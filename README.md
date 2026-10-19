# BoundCertain

**Boundary-Aware Semantic Segmentation with Predictive Uncertainty Estimation**

BoundCertain is a deep learning project for pixel-level scene understanding in street images. It combines boundary-aware feature fusion with uncertainty estimation to improve scene segmentation and help identify potentially unreliable predictions.

Kaggle notebook : https://www.kaggle.com/code/jesvanthsabarishvk/certainsemseg

## Key Features

- **Custom segmentation architecture:** ConvNeXt-Base encoder with a dedicated boundary prediction branch.
- **Gated boundary fusion:** Uses learned gating to integrate boundary features into segmentation features.
- **Hybrid training objective:** Combines Lovász-Softmax, focal loss, boundary supervision, and auxiliary supervision.
- **Uncertainty estimation:** Uses Monte Carlo dropout to estimate predictive uncertainty.
- **Test-time augmentation (TTA):** Supports augmented inference for final predictions.

## Dataset

The model is trained and evaluated on the **CamVid street-scene dataset**, using 11 semantic classes:

Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, and Bicyclist.

The notebook uses 369 training images and 100 validation images.

## Results

| Metric | Validation Result |
|---|---:|
| Mean Intersection over Union (mIoU) | 80.32% |
| Pixel Accuracy | 95.73% |

*Results are from the notebook's final evaluation with test-time augmentation.*

## Tech Stack

- Python
- PyTorch
- timm
- Albumentations
- OpenCV
- NumPy and Matplotlib

## Running the Project

The original notebook was configured for Kaggle with the CamVid dataset available under `/kaggle/input/`.

1. Open the notebook in Kaggle.
2. Attach the CamVid dataset.
3. Install the required dependencies:

   ```bash
   pip install timm==0.9.12 albumentations==1.3.1 segmentation-models-pytorch
   ```

4. Run the notebook cells in order to configure the dataset, train the model, evaluate validation performance, and generate prediction visualizations.

The notebook saves the best model checkpoint as `best_model.pth` and prediction visualizations as `preds.png`.

## Project Objective

The project explores whether explicit boundary supervision and predictive uncertainty estimation can support more informative semantic segmentation of complex street scenes.

**Note:** The reported metrics describe this implementation's validation results; they do not establish improvement over a baseline without a controlled comparison.

## Future Work

- Compare against a baseline segmentation model.
- Evaluate performance separately for each semantic class.
- Investigate boundary accuracy and uncertainty calibration.
- Benchmark inference latency and memory usage.
- Extend evaluation to the BDD100K dataset (https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) to assess the model's generalization across diverse driving environments, lighting conditions, and weather in real-world street-scene segmentation.
