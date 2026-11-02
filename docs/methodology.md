# Methodology

## Dataset

PneumoniaMNIST is loaded through MedMNIST with its official `train`, `val`, and `test` splits. The API exposes class counts and samples by split. The test partition is never included in training or model selection.

## Conditional GAN

The generator concatenates a learned class embedding with latent noise and decodes it to a 28 × 28 grayscale image using transposed convolutions and a `Tanh` output. The discriminator applies convolutional features to the image and adds a learned label projection. The two networks are optimized separately with binary cross-entropy and Adam. A checkpoint and sample grid are saved per completed epoch.

## Classifier comparison

The same small CNN architecture is used for three training sets: original data, original data with mild geometric transforms, and original data with cGAN samples added for the minority class. Validation loss chooses the checkpoint. The selected checkpoint is evaluated on the same official test split and produces accuracy, macro precision, macro recall, macro-F1, a confusion matrix, and train/validation loss histories.

Results are empirical and do not imply that synthetic augmentation improves performance. A stronger academic claim should repeat runs across seeds and include a prespecified statistical analysis.

## Repeated-seed comparisons

Experiment Lab can run one, three, or five repeats, starting at seed 42 and incrementing by one. Each seed value is applied to every selected variant so results can be compared by seed. Runs are grouped by a batch identifier; the dashboard summarizes completed test metrics within the latest batch using the arithmetic mean and sample standard deviation. Incomplete and blocked runs are excluded from those summaries and remain visible in experiment history. These descriptive statistics do not replace a prespecified inferential analysis.
