# IonoBench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnonPaperReview/DemoRepo/blob/main/tutorial/colab_version.ipynb?flush_cache=true)
[![Hugging Face Dataset](https://img.shields.io/badge/HF%20Datasets-IonoBenchv1-blue?logo=huggingface)](https://huggingface.co/datasets/Mertjhan/IonoBenchv1)
[![Hugging Face Models](https://img.shields.io/badge/HF%20Models-IonoBenchv1-blue?logo=huggingface)](https://huggingface.co/Mertjhan/IonoBenchv1)

**IonoBench**: Evaluating Spatiotemporal Models for Ionospheric Forecasting under Solar-Balanced and Storm-Aware Conditions  
*Accepted in Remote Sensing (MDPI), Special Issue on Ionosphere and Space Weather*

---

**IonoBench** is a benchmark framework for evaluating deep spatiotemporal models on ionospheric Total Electron Content (TEC) forecasting.  
It includes standardized datasets, evaluation protocols, pretrained models, and configuration-based experimentation.

This repository provides the full IonoBench framework (under development).  
Click **Open in Colab** to test without local setup.

---

### Features

- Stratified and chronological TEC datasets
- Model registry and configuration system
- Pretrained model download via Hugging Face
- Solar-balanced and storm-aware evaluation experiments
- Colab tutorial for quick testing

---

### Framework Status

| Component               | Status      |
|------------------------|-------------|
| HF model & data access | Complete    |
| Model/config registry  | Complete    |
| Evaluation pipeline    | Complete    |
| Visualization tools    | In Progress |
| CLI support            | Planned     |
| Training tutorials     | Planned     |
| Contributor guide      | Planned     |

---

### Local Setup (Python 3.11 + CUDA)

```bash
# Create environment
conda create -n ionobench python=3.11 -y
conda activate ionobench

# Install dependencies
pip install -r requirements.txt

```

This environment uses `torch==2.5.1`, which requires a compatible CUDA build.  
PyTorch provides separate wheels for each CUDA version (e.g., `+cu118`, `+cu121`, `+cu124`).  
Make sure your NVIDIA driver supports the CUDA version included in the installed wheel.  

To install the correct version, refer to the official selector:  
https://pytorch.org/get-started/locally/