# DIT5411 Handwritten Chinese Character Recognition (13,065 Classes)

This project trains and compares three convolutional neural networks (CNNs) in TensorFlow/Keras to recognise 13,065 handwritten Traditional Chinese characters. The three models are compared on a held-out test set.

## Final Results

All three models use the same data, augmentation and training budget (2 epochs each). For each model, the epoch with the highest validation accuracy was kept. The test set was evaluated only once, at the very end.

| Model | Parameters | Validation accuracy | **Test accuracy** | Test loss |
|---|---:|---:|---:|---:|
| `C_residual` | 5,484,249 | 92.09% | **90.92%** | 0.3859 |
| `B_deeper` | 5,472,537 | 90.33% | 88.86% | 0.4896 |
| `A_spatial` | 5,460,825 | 83.59% | 80.81% | 0.8505 |

- Test set: 162,817 images covering all 13,065 classes.
- Random-guess accuracy is about 1 / 13,065 ≈ 0.0077%.
- For all three models, the best epoch was the last one (epoch 2), so they may not have fully converged.

## Dataset

- Source: [chenkenanalytic/handwritting_data_all](https://github.com/chenkenanalytic/handwritting_data_all). This project uses the full dataset, not the 4,803-character common-words version.
- Size: 13,065 class folders, 684,677 PNG images in total (black handwriting on a white background, 300 × 300 pixels).
- Each folder name is a class ID. The mapping from model output index to folder ID is saved in `labels.json`.
- The dataset is **not** included in this repository because of its size. See "How to Reproduce" below.

## Data Split

1. **Train/test split:** within each class, files are sorted by filename. The first 40 images are used for training and the remaining images are used for testing.
2. **Exception for small classes:** 43 classes have 40 or fewer images (the smallest has 16). To keep all 13,065 classes and still give every class at least one test image, these 43 classes use the first 80% of their images for training and the rest for testing. Details are in `split_exceptions.csv`.
3. **Validation set:** about 10% of each class's training images are randomly held out as a validation set. It is used to choose the best epoch and to monitor training. The test set was never used for any tuning decision.

| Split | Original images |
|---|---:|
| Training | 469,673 |
| Validation | 52,187 |
| Test | 162,817 |
| **Total** | **684,677** |

## Preprocessing and Data Augmentation

- Images are read as grayscale, resized to 64 × 64 with bilinear interpolation, and scaled to the 0–1 range (divided by 255).
- **200 training samples per class per epoch:** each class's training images are resampled cyclically until the class has 200 samples, and each sample is then randomly augmented. One epoch therefore contains 13,065 × 200 = 2,613,000 samples (81,657 batches with batch size 32). Several samples can come from the same original image, with a different random shift each time.
- **Augmentation:** random horizontal and vertical translation of at most 1 pixel in each direction (`RandomTranslation(1/64)`). Uncovered edges are filled with white, and nearest-neighbour interpolation is used so that strokes stay sharp.
- Validation and test images are not augmented.
- Training indices are shuffled globally before images are loaded, so each batch contains a wide mix of classes.

## Model Architectures

All three models share the same input and classification head:

```
Input 64×64×1 → convolutional blocks → Flatten
→ Dense(256) + BatchNormalization + ReLU → Dense(13,065, softmax)
```

Every convolution is 3 × 3 with `padding="same"`, followed by BatchNormalization and ReLU. There are two convolutional blocks with 16 and 32 filters, and each block ends with 2 × 2 max pooling. The models differ only inside each block:

| Model | Convolutions in each block |
|---|---|
| `A_spatial` | 1 convolution (shallow baseline) |
| `B_deeper` | 2 stacked convolutions (deeper) |
| `C_residual` | 1 convolution, plus a residual branch of 2 convolutions that is added back to the main path before a ReLU |

## Training Setup

- Environment: Windows 11 with WSL 2 (Ubuntu), TensorFlow installed as `tensorflow[and-cuda]`, NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB).
- Optimiser: Adam, learning rate 0.001. Loss: sparse categorical cross-entropy. Batch size 32. 2 epochs per model. Random seed 42.
- Each epoch runs: **train → recalibrate BatchNormalization → validate → save**. The epoch with the highest validation accuracy is kept as the best model.
- **BatchNormalization recalibration:** after each training epoch, 16,000 un-augmented training images (500 batches) are passed through the model in training mode. This only updates the BatchNormalization moving averages; the weights do not change. Only training data is used for this step.

## Problems Found During Development

The final pipeline came out of several rounds of diagnosis. Each change below was supported by a controlled experiment.

| Problem | Evidence | Fix |
|---|---|---|
| The original strong augmentation (rotation, ±10% zoom, ±5% translation) prevented learning | 200-class experiment: validation accuracy was 3.12% with the original augmentation, 64.25% with no augmentation, and 74.06% with a 1-pixel shift | Switched to a 1-pixel random translation |
| The first models did not learn at all on the full dataset | The first version reached only 0.01% test accuracy (close to random). All 64 features before the output layer were zero (dead ReLU units) | Added BatchNormalization and widened the dense layer before the output from 64 to 256 units |
| BatchNormalization moving averages lagged behind the changing weights, so inference-mode predictions were inaccurate | After 10,000 batches, loss on 1,024 validation images was 6.61 in inference mode and fell to 3.04 after recalibration | Recalibrate BatchNormalization before every validation |
| Images were stored in class order, and a local shuffle did not mix them enough | The shuffle buffer (50,000) was much smaller than the ~520,000 training images | Shuffle integer indices globally before loading images |

## Limitations

- Each model was trained for only 2 epochs, and the best epoch was always the last one. Longer training would likely improve accuracy further.
- The 43 small classes use a different split rule from the other classes.
- The 200 training samples per class are produced by resampling a limited number of original images with random shifts, so they are not 200 distinct handwriting samples.

## Repository Contents

```
.
├── README.md
├── requirements.txt
├── ML_assignment.ipynb          # final pipeline: data split, augmentation, models, training, testing
└── results/
    ├── test_results.csv         # final test scores
    ├── run_config.json          # training configuration
    ├── labels.json              # output index → class folder ID
    ├── split_exceptions.csv     # the 43 small classes and how they were split
    ├── A_spatial_state.json     # per-epoch training/validation history
    ├── B_deeper_state.json
    └── C_residual_state.json
```

## How to Reproduce

1. Use Linux or WSL 2 with an NVIDIA GPU, then install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download and extract the full dataset from [chenkenanalytic/handwritting_data_all](https://github.com/chenkenanalytic/handwritting_data_all) into `data/cleaned_data/`.
3. Open `ML_assignment.ipynb` in JupyterLab, set `QUICK_TEST = False`, and run the cells in order. Training all three models for 2 epochs took about 4 hours on an RTX 3050 Ti Laptop GPU.
