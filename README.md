# Comparative Deepfake Video Detection Using EfficientNet-B0 and Vision Transformer

A reproducible MSc research project for **deepfake video detection** comparing **EfficientNet-B0** and **Vision Transformer (ViT-Base Patch16-224)** under the same video-level evaluation pipeline, with a separate external dataset used as an exploratory stress test.

> **Author:** Pusa Prashanth  
> **Programme:** MSc Data Science and Artificial Intelligence  
> **Institution:** Sheffield Hallam University  
> **Supervisor:** Salem Mansour  
> **Repository:** https://github.com/pusaprashanth6/deepfake-detection-vit-vs-efficientnet

---

## Project Overview

Deepfake video detection can produce misleadingly strong results when evaluation protocols allow correlated frames from the same source video to appear across training and test data. Performance can also decrease when a detector is evaluated on media from a different dataset or generation process.

This project therefore focuses on **evaluation validity and reproducibility**, rather than proposing a new state-of-the-art architecture.

The study compares:

- **EfficientNet-B0** — a compact convolutional neural network
- **ViT-Base Patch16-224** — a Vision Transformer using patch-based self-attention

Both models are trained and evaluated using the same preprocessing, video partitions and metric definitions.

---

## Research Question

**How accurately and generalisably can EfficientNet-B0 and ViT-Base classify real and manipulated videos using video-level partitions and a separate external dataset?**

---

## Main Objectives

1. Audit dataset provenance, licensing and research ethics requirements.
2. Create a balanced and reproducible video-level experimental dataset.
3. Prevent identical video files from crossing train, validation and test partitions.
4. Train and compare EfficientNet-B0 and ViT-Base under the same pipeline.
5. Evaluate both models using frame-level and video-level metrics.
6. Perform an exploratory external-dataset stress test.
7. Use Grad-CAM to inspect EfficientNet-B0 predictions qualitatively.
8. Preserve reproducible artefacts such as manifests, training histories, predictions and evaluation outputs.

---

## Datasets

### 1. Development Dataset

**Deep Fake Detection (DFD) Entire Original Dataset**

Kaggle:  
https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data

The dataset contains original and manipulated video sequences derived from the FaceForensics dataset family.

The notebook discovered:

- **3,431 total videos**
- **3,068 Fake**
- **363 Real**

For the controlled experiment, a balanced sample was created:

- **300 Real videos**
- **300 Fake videos**
- **600 videos total**

### 2. External Evaluation Dataset

**Deepfake Testing Videos**

Kaggle:  
https://www.kaggle.com/datasets/chandashekar8/deepfake-testing-videos

The external source was kept completely separate from training, checkpoint selection, model selection and threshold tuning.

Only **2 supported labelled videos** were available in the final external evaluation, so the result is treated as **exploratory rather than statistically conclusive**.

---

## Experimental Workflow

```text
DFD Dataset
    |
    v
Dataset Audit and Label Validation
    |
    v
Balanced Sample: 300 Real + 300 Fake
    |
    v
Video-Path Split Before Frame Extraction
    |
    +---- Train: 420 videos
    +---- Validation: 90 videos
    +---- Test: 90 videos
    |
    v
16 Uniformly Sampled Frames per Video
    |
    v
Face Localisation with OpenCV Haar Cascade
    |
    +---- Detected face crop
    +---- Centre-crop fallback when no face is detected
    |
    v
Resize to 224 x 224 + ImageNet Normalisation
    |
    +-------------------------+
    |                         |
    v                         v
EfficientNet-B0          ViT-Base Patch16-224
    |                         |
    +------------+------------+
                 |
                 v
Validation-Only Checkpoint and Threshold Selection
                 |
                 v
Frame-Level and Video-Level Evaluation
                 |
                 v
External Dataset Stress Test
                 |
                 v
Error Analysis + Grad-CAM
```

---

## Preprocessing

The final preprocessing pipeline includes:

1. Dataset discovery and label audit
2. Balanced sampling
3. Video-path split before frame extraction
4. Uniform frame sampling
5. Face localisation using OpenCV Haar cascade
6. Approximate face-margin expansion
7. Centre-crop fallback when no face box is detected
8. Resize to **224 x 224**
9. ImageNet normalisation
10. Training-time augmentation
11. Horizontal-flip test-time augmentation during inference

### Frame Extraction

- **16 frames per video**
- **600 videos processed**
- **9,600 total extracted frames**

### Face Localisation

Face detection succeeded on approximately:

- **71.17% of frames**

Centre-crop fallback was used on:

- **28.83% of frames**

This is an important limitation because a fallback crop may contain less useful facial information.

---

## Models

### EfficientNet-B0

EfficientNet is a convolutional neural-network family that scales network depth, width and input resolution in a coordinated way.

Why it was selected:

- Strong transfer-learning baseline
- Efficient computational cost
- Suitable for local textures and blending artefacts
- Practical for MSc-scale GPU resources

### Vision Transformer — ViT-Base Patch16-224

ViT divides an image into fixed-size patches and processes them using transformer self-attention.

Why it was selected:

- Provides a substantially different architecture from a CNN
- Can model long-range relationships across image regions
- Enables a controlled **CNN vs Transformer** comparison

---

## Training Strategy

The final notebook uses:

- ImageNet pretrained weights
- Classifier warm-up
- Full fine-tuning
- AdamW optimisation
- Focal cross-entropy
- Data augmentation
- Early stopping
- Best-validation checkpoint restoration
- Validation-only threshold selection

### Best Checkpoints

| Model | Best checkpoint | Best validation frame ROC-AUC |
|---|---:|---:|
| EfficientNet-B0 | Fine-tune epoch 12 | 0.8438 |
| ViT-Base Patch16-224 | Fine-tune epoch 8 | 0.7307 |

### Validation Video ROC-AUC

| Model | Validation video ROC-AUC |
|---|---:|
| EfficientNet-B0 | **0.8894** |
| ViT-Base Patch16-224 | **0.7793** |

EfficientNet-B0 was therefore selected before the internal test set was evaluated.

---

## Video-Level Inference

For each frame:

1. Predict Fake probability on the original image.
2. Predict again using a horizontal flip.
3. Average both probabilities.

For each video:

1. Sort the 16 frame probabilities.
2. Remove the **2 lowest** and **2 highest** values.
3. Average the remaining **12 probabilities**.
4. Compare the final video probability with the frozen validation threshold.

### Frozen Thresholds

| Model | Validation threshold |
|---|---:|
| EfficientNet-B0 | **0.5202** |
| ViT-Base Patch16-224 | **0.5783** |

---

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Average Precision (AP)
- Equal Error Rate (EER)
- Confusion Matrix

Both **frame-level** and **video-level** results are reported.

Video-level performance is treated as the primary result because the independent experimental unit is the video rather than an individual correlated frame.

---

## Final Results

### Internal Test Performance

| Model | Level | Accuracy | Recall | F1 | ROC-AUC | AP | EER |
|---|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | Frame | 0.7410 | 0.8042 | 0.7564 | 0.8431 | 0.8536 | 0.2535 |
| ViT-Base | Frame | 0.6285 | 0.4403 | 0.5423 | 0.6991 | 0.7197 | 0.3799 |
| **EfficientNet-B0** | **Video** | **0.8000** | **0.8222** | **0.8043** | **0.8667** | **0.8553** | **0.2000** |
| ViT-Base | Video | 0.6778 | 0.4667 | 0.5915 | 0.7595 | 0.7714 | 0.3222 |

### Internal Confusion-Matrix Interpretation

EfficientNet-B0:

- **72 / 90 videos classified correctly**
- **37 / 45 Fake videos correctly detected**
- **8 / 45 Fake videos missed**

ViT-Base:

- **61 / 90 videos classified correctly**
- **21 / 45 Fake videos correctly detected**
- **24 / 45 Fake videos missed**

The main internal finding is that **EfficientNet-B0 is the stronger model under this experimental protocol**.

---

## External Stress Test

The external evaluation contained only:

- **1 Real video**
- **1 Fake video**

EfficientNet-B0 predicted both as Real.

External results:

- Accuracy: **50%**
- Fake Recall: **0**
- F1-score: **0**
- AP: **0.5000**
- EER: **1.0000**

This result may indicate domain shift, but the sample size is too small to support a statistically reliable generalisation claim.

Therefore, the project concludes:

> **EfficientNet-B0 is the stronger internal detector, but robust cross-dataset generalisation remains unproven.**

---

## Explainability with Grad-CAM

Grad-CAM was used to inspect EfficientNet-B0 predictions.

The analysis showed that:

- Some predictions focus on facial regions.
- Some predictions also activate on background or contextual areas.
- Background attention may indicate shortcut learning.

Grad-CAM is used only as a **diagnostic visualisation**. It is not treated as causal proof of the model's reasoning.

---

## Repository Structure

```text
deepfake-detection-vit-vs-efficientnet/
|
|-- README.md
|-- deepfake-detection.ipynb
|-- deepfake_flask_app.zip
|
|-- efficientnet_b0_training_history.csv
|-- vit_base_patch16_224_training_history.csv
|
|-- video_manifest.csv
|-- frame_manifest.csv
|-- frame_extraction_log.csv
|
|-- external_manifest_review.csv
|-- external_frame_manifest.csv
|-- external_frame_extraction_log.csv
|-- external_predictions.csv
|-- external_metrics.csv
```

> The exact file list may expand as additional generated outputs are uploaded.

---

## Running the Notebook

### Recommended Environment

The notebook was designed for a GPU-enabled Python environment such as Kaggle.

Typical dependencies include:

```bash
pip install torch torchvision timm opencv-python scikit-learn pandas numpy matplotlib tqdm pillow
```

### Kaggle Dataset Paths

Development dataset:

```text
/kaggle/input/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
```

External dataset:

```text
/kaggle/input/datasets/chandashekar8/deepfake-testing-videos
```

### Execution Order

Run the notebook from top to bottom.

The main stages are:

1. Import packages and configuration
2. Discover videos and create manifest
3. Audit class distribution
4. Create balanced sample
5. Split at video-path level
6. Extract and preprocess frames
7. Build PyTorch datasets and dataloaders
8. Train EfficientNet-B0
9. Train ViT-Base
10. Select checkpoints and thresholds
11. Evaluate frame-level performance
12. Aggregate predictions to video level
13. Compare internal test results
14. Run external stress test
15. Generate error analysis and Grad-CAM outputs

---

## Reproducibility Notes

Important controls used in the final experiment:

- Split performed before frame extraction
- No identical video path is shared across train, validation and test sets
- External data are not used for training or threshold tuning
- Model selection is validation-based
- Internal test results are produced only after model selection
- Results are reported at both frame and video level
- The project uses one final experimental run, so uncertainty across repeated seeds is not estimated

### Leakage Caveat

The project verifies **video-path disjointness**.

It does **not** claim that all related actors or original/manipulated source pairs were grouped before splitting.

For that reason, the correct description is:

> **Path-disjoint video-level split**

rather than a claim of complete dataset-level leakage elimination.

---

## Ethical and Data-Governance Considerations

The project uses secondary video data containing identifiable facial imagery.

Controls include:

- UREC2 ethics route
- No participant recruitment
- No surveys, interviews or user testing
- No identity recognition
- No demographic profiling
- No raw facial-video redistribution through this repository
- Dataset access and licence evidence retained separately
- Privacy-aware reporting of visual evidence
- Aggregate model results used wherever possible

The raw datasets must be downloaded from their original providers and are **not distributed in this repository**.

---

## Optional Local Demonstration

The repository may contain:

```text
deepfake_flask_app.zip
```

This is a researcher-only local demonstration artefact.

It is **not** the source of the reported research metrics and was not used for participant testing or public deployment.

The reported results come from the final notebook evaluation pipeline.

---

## Limitations

The main limitations are:

- Only 600 of the 3,431 DFD videos were used in the controlled experiment
- Only one final training run was used
- No repeated-seed confidence intervals were calculated
- Related actor/source-pair separation was not fully verified
- 28.83% of frames used centre-crop fallback
- Training curves showed overfitting
- External evaluation contained only 2 videos
- Robust cross-dataset generalisation is therefore not established

---

## Future Work

Future improvements could include:

- Use the complete or a larger DFD training set
- Run multiple random seeds
- Report confidence intervals
- Group original/manipulated source pairs before splitting
- Replace Haar cascade with RetinaFace or MTCNN
- Add automatic face-crop quality filtering
- Evaluate on larger independent external datasets
- Add probability calibration
- Explore temporal models
- Explore frequency-domain deepfake features
- Compare self-supervised and domain-generalisation methods

---

## Key References

- Dosovitskiy, A., et al. (2021). *An image is worth 16x16 words: Transformers for image recognition at scale*. ICLR.
- Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). *Celeb-DF: A large-scale challenging dataset for deepfake forensics*. CVPR.
- Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). *FaceForensics++: Learning to detect manipulated facial images*. ICCV.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. ICCV.
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. ICML.

---

## Author

**Pusa Prashanth**  
MSc Data Science and Artificial Intelligence  
Sheffield Hallam University

Supervisor: **Salem Mansour**

GitHub:  
https://github.com/pusaprashanth6/deepfake-detection-vit-vs-efficientnet

---

## Academic Context

This repository supports an MSc research project and is intended for academic reproducibility, evaluation and demonstration.

The project does not claim that the trained model is suitable for production deployment or that it can reliably detect all forms of deepfake media.

---

## Final Project Conclusion

**EfficientNet-B0 achieved the strongest internal performance in the final experiment, with 80.00% video-level accuracy and 0.8667 ROC-AUC. ViT-Base achieved 67.78% video-level accuracy and 0.7595 ROC-AUC. The external two-video stress test was insufficient to establish robust cross-dataset generalisation.**
