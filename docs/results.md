# Results

These are the results from the local MVP run on **2026-11-01**. They are actual measurements from one seed, not estimates. The run used PyTorch `2.15.0+cpu`, seed `42`, batch size `64`, and five epochs per classifier variant. The official test partition contains 624 images (234 Normal, 390 Pneumonia); it was held out from training and validation-based checkpoint selection.

| Configuration | Accuracy | Macro precision | Macro recall | Macro-F1 | Confusion matrix |
|---|---:|---:|---:|---:|---|
| Baseline CNN | 88.14% | 90.02% | 85.04% | 86.63% | `[[170, 64], [10, 380]]` |
| Conventional augmentation CNN | 75.64% | 85.42% | 67.61% | 67.93% | `[[83, 151], [1, 389]]` |
| GAN-augmented CNN | 78.04% | 86.10% | 70.90% | 72.05% | `[[99, 135], [2, 388]]` |

Confusion matrix order is `[Normal, Pneumonia]` for both rows and columns. In this single run, GAN augmentation scored higher macro-F1 than conventional augmentation, while the baseline scored higher than both. This result does **not** support claiming that synthetic augmentation improved the baseline. Small benchmark images, five-epoch training, one random seed, and CPU execution limit what can be concluded.

## cGAN pipeline run

The conditional GAN completed five epochs on the official training split, processing 4,672 images per epoch (`drop_last=True`). Recorded generator/discriminator losses were:

| Epoch | Generator loss | Discriminator loss |
|---:|---:|---:|
| 1 | 1.6844 | 0.4456 |
| 2 | 2.4639 | 0.3909 |
| 3 | 1.1827 | 0.5348 |
| 4 | 0.9774 | 0.6163 |
| 5 | 0.9316 | 0.6318 |

The saved checkpoint generated four Normal and four Pneumonia samples through the API; the experiment added 2,280 generator-produced Normal images to balance its training data. These synthetic images are experiment artifacts, not additional real labels.

## Reproduction and stronger evidence

Rerun all variants from Experiment Lab with more epochs and repeat with several seeds. Preserve each run in the database. For a stronger study, predefine a validation strategy, compare distributions across seeds, quantify image diversity/quality, and report uncertainty. Do not use these results for patient care or clinical decisions.
