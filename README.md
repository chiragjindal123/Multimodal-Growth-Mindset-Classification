# Multimodal Growth Mindset Classification
 
This repository contains the dataset metadata and deep learning codebase for our research on classifying **Growth Mindset** from structured video interviews.

By leveraging a rigorous **sentence-level hierarchical modeling** approach and **multimodal sensor fusion** (Text, Audio, Facial Expressions, Heart Rate Variability, and Eye Movement), this project automatically classifies participants into distinct mindset categories based on **Dweck’s Implicit Theories of Intelligence**.

---

## 📊 Dataset & Annotations

The data consists of multimodal features extracted from structured interviews with **98 participants**.

- **High-Fidelity Filtering:**  
  To ensure the model learns from rich psychological signals rather than conversational noise, all transcripts underwent strict **human-in-the-loop re-annotation**.  
  Utterances lacking semantic value (e.g., conversational fillers, backchanneling) were explicitly tagged as `-1` and programmatically filtered out before training.

- **Safe Data Included:**  
  Anonymized label metadata (e.g., `output_0704_redefined.csv`) is provided to demonstrate dataset distribution and filtering logic.

> **Note:** Raw large-scale `.npy` arrays for continuous face/audio/eye tracking are omitted from this repository due to size constraints.

---

## ⚙️ Environment & Hardware

- **Deep Learning Framework:** PyTorch  
- **Machine Learning Metrics:** scikit-learn (10-Fold Stratified Cross-Validation)  
- **Hardware:** NVIDIA TITAN RTX GPU (used for all feature extraction and model training)

---

## 📂 Repository Structure

- `train.py`  
  Master PyTorch training script. Includes model architecture, dataset routing, and comprehensive 10-Fold Cross-Validation loop for automated ablation studies.

- `preprocessing.py`  
  Handles strict multimodal alignment, maps exact human labels to sentence context, and enforces facial time-series truncation/padding to **120 frames**.

- `csvedit.py` / `readjson.py`  
  Manages label processing, merging, and computation of final subject-level mindset scores (averages and modes).

- `check.py`  
  Validation script to verify that all 5 modalities (Text, Audio, Face, HRV, Eye) are perfectly aligned across participants.

- `10Fold_Ablation_Results.txt`  
  Raw output logs of the full ablation study.

---

## 🚀 How to Run the Pipeline

### Step 1: Multimodal Alignment & Preprocessing

Run preprocessing to parse raw JSON, align all 5 sensors to transcribed sentences, and drop missing data.

```bash
python preprocessing.py
```

### Step 2: Validation Check

Ensure all modality arrays align correctly before training.

```bash
python check.py
```

### Step 3: Model Training & Ablation Study

Run the master training loop. This automatically generates valid sensor combinations and evaluates them with Stratified 10-Fold Cross-Validation.

```bash
python train.py
```

---



---

## 📌 Summary

This work demonstrates that careful multimodal curation, psychologically informed annotation, and hierarchical sentence-level learning can improve growth mindset classification from interview data—while highlighting that **more modalities are not always better** for robust F1 performance.
