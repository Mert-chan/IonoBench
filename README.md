# IonoBench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mert-chan/IonoBench/blob/main/tutorial/colab_version.ipynb?flush_cache=true)
[![Hugging Face Dataset](https://img.shields.io/badge/HF%20Datasets-IonoBench-blue?logo=huggingface)](https://huggingface.co/datasets/Mertjhan/IonoBench)
[![Hugging Face Models](https://img.shields.io/badge/HF%20Models-IonoBench-blue?logo=huggingface)](https://huggingface.co/Mertjhan/IonoBench)

**IonoBench**: Evaluating Spatiotemporal Models for Ionospheric Forecasting under Solar-Balanced and Storm-Aware Conditions  
*Accepted in Remote Sensing (MDPI), Special Issue on Ionosphere and Space Weather*

---

**IonoBench** is a benchmark framework for evaluating deep spatiotemporal models on global Total Electron Content (TEC) forecasting.  
It includes standardized datasets, evaluation protocols, pretrained models, and configuration-based experimentation.
  
Click **Open in Colab** to test without local setup.

---

### Features
- Supports **multichannel spatiotemporal models** for multistep 24-hour input → 24-hour output setup
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

### Local Setup 

```bash
# Clone repository
git clone https://github.com/Mert-chan/IonoBench.git
```
```base
# Change your directory
cd IonoBench
```
```bash
# Create environment
conda create -n ionobench python=3.11 -y
conda activate ionobench
```
```bash
# Install dependencies
pip install -r requirements.txt
```
Tested on: **Python 3.11.13 + PyTorch 2.5.1 + CUDA 12.4** //
This environment uses `torch==2.5.1`, which requires a compatible CUDA build.  
PyTorch provides separate wheels for each CUDA version (e.g., `+cu118`, `+cu121`, `+cu124`).  
Make sure your NVIDIA driver supports the CUDA version included in the installed wheel.  
