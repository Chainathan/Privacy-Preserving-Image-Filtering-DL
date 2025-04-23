# 📷 Privacy-Preserving Image Filtering – Model Training Repository

This repository contains all model training, evaluation, and dataset preparation pipelines used to support the [Perception Privacy Mobile App](https://github.com/Chainathan/Perception-Privacy-Mobile-App). The models developed here are optimized for detecting and masking sensitive content in both documents and general images before sharing.

---

## 📦 Model Stack

| Task              | Model                  |
| ----------------- | ---------------------- |
| Segmentation      | YOLOv11 / YOLOv8 / SAM |
| Text Analysis     | GPT-3.5 / MobileBERT   |
| OCR               | Google ML Kit          |
| Mobile Conversion | TFLite                 |

---

## 🔍 Repository Overview

### 🧠 Segmentation Training

- Instance segmentation training scripts using YOLOv8 and DeepLabV3.
- Trained to detect sensitive objects: ID cards, screens, QR codes, and license plates.
- Incorporates weakly-supervised learning using pseudo masks from bounding boxes.

### 🧪 Segmentation Testing

- Evaluation scripts for mAP, IoU, precision, recall.
- Supports benchmarking across individual classes or combined masks.
- Includes model export pipeline to TensorFlow Lite for mobile deployment.

### 🗺️ Segmentation Mask Generation

- Converts bounding box-only datasets to segmentation masks.
- Techniques used:
  - SAM2 model for generating segmentation masks with bounding box prompts.
- Output formats: COCO and YOLO.

### 📦 Dataset Conversion Tools

- Convert VOC, YOLO, and custom annotations to COCO format.
- Create unified datasets from diverse labeling schemas.
- Validate and visualize annotation integrity.

### 🏃 MobileBERT Training for Named Entity Recognition (PII)

- Train MobileBERT for sensitive content classification in OCR-extracted text.
- Optimized for detecting:
  - Names, dates, phone numbers, SSNs, credit cards, emails, addresses.

### 🧪 GPT API Testing

- Scripts for testing OpenAI GPT-4 and GPT-3.5 for PII NER.

---

## 🔄 Workflow Integration

This repository supports the full training and deployment pipeline:

1. **Prepare Datasets**: Convert and unify annotations.
2. **Generate Segmentation Masks**: For incomplete datasets.
3. **Train Segmentation Model**: For image-based privacy filters.
4. **Train Text Classifier**: For OCR-based PII detection.
5. **Test & Export Models**: Benchmark and export to TFLite/ONNX.
6. **Deploy to App**: Models are used in [Perception Privacy Mobile App](https://github.com/Chainathan/Perception-Privacy-Mobile-App).

---

## 🧪 Datasets

The Instance Segmentation model is trained using **three distinct datasets**, each containing annotations for a specific sensitive class:

| Class            | Label Type            |
| ---------------- | --------------------- |
| ID Cards         | Instance Segmentation |
| License Plates   | Bounding Boxes        |
| Computer Screens | Bounding Boxes        |

- All datasets are converted to **COCO format**.
- Bounding boxes are converted to pseudo-masks using **SAM2**.

---
