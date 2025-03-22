# 📷 Privacy-Preserving Image Filtering Mobile App

A mobile application that automatically detects and censors sensitive content in images — whether documents or everyday photos — enabling safe and private sharing. The app uses OCR, LLMs, and segmentation models to locate and mask content such as ID cards, credit cards, screens, license plates, QR codes, and confidential text.

---

## Status

⚠️ **Note:** This project is currently under active development.

---

## 🚀 Features

- 🔍 **Image Type Classification**  
  Automatically classifies uploaded images as **documents** or **generic photos**.

- 🧾 **Document Processing**

  - Applies OCR to extract text.
  - Uses a Large Language Model (LLM) to detect sensitive content.
  - Maps sensitive content back to the image using OCR coordinates or layout models.
  - Censors sensitive text with blur or blackout filters.

- 🖼️ **Generic Image Processing**

  - Performs instance segmentation to detect:
    - License Plates
    - ID/Passport/Credit Cards
    - Computer Screens
    - QR Codes
  - Uniformly censors detected objects with black bars or blur.

- 🛡️ **Custom Privacy Filters**

  - Choose between **blur**, **black box** or **image inpainting** censoring.
  - Preview and adjust before saving or sharing.

- 📱 **Mobile Deployment**
  - Lightweight, real-time models optimized for on-device inference (TFLite/ONNX).
  - Supports Android and iOS.

---

## 🧠 Architecture Overview

1. **Image Classifier** – Distinguishes between document and generic images.
2. **Document Mode**
   - OCR → LLM → Text Censorship Mapping
3. **Generic Image Mode**
   - Instance segmentation over combined datasets
   - Unified mask generation for all sensitive object types

---

## 🧪 Datasets

The app is trained using **four distinct datasets**, each containing annotations for a specific sensitive class:

| Class            | Label Type            |
| ---------------- | --------------------- |
| ID Cards         | Instance Segmentation |
| License Plates   | Bounding Boxes        |
| Computer Screens | Bounding Boxes        |

- All datasets are converted to **COCO format**.
- Bounding boxes are converted to pseudo-masks using techniques like **GrabCut**, **SAM**, or **Mask R-CNN**.

---

## 📦 Model Stack

| Task                 | Model                        |
| -------------------- | ---------------------------- |
| Image Classification | MobileViT / EfficientNet     |
| OCR                  | TrOCR / Tesseract            |
| Text Analysis        | GPT-4 / LayoutLM / Donut     |
| Segmentation         | Mask R-CNN / DeepLabV3 / SAM |
| Mobile Conversion    | TFLite / ONNX                |

---
