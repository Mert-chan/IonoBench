# IonoBench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mert-chan/IonoBench/blob/main/tutorial/colab_version.ipynb?flush_cache=true)
[![Hugging Face Dataset](https://img.shields.io/badge/HF%20Datasets-IonoBench-blue?logo=huggingface)](https://huggingface.co/datasets/Mertjhan/IonoBench)
[![Hugging Face Models](https://img.shields.io/badge/HF%20Models-IonoBench-blue?logo=huggingface)](https://huggingface.co/Mertjhan/IonoBench)

**IonoBench**: Evaluating Spatiotemporal Models for Ionospheric Forecasting under Solar-Balanced and Storm-Aware Conditions  
*Accepted in Remote Sensing (MDPI)*

---

This project is a benchmark framework for evaluating deep spatiotemporal models on Global Ionospheric Map (GIM) forecasting. The framework provides standardized datasets, evaluation protocols, pretrained models, and configuration-based experimentation.

### Overall Performance on IonoBench Test Set
| Model       | RMSE (↓)       | R² (↑)         | SSIM (↑)       |
|-------------|----------------|----------------|----------------|
| **SimVPv2** | **2.25 ± 1.35** | **0.962 ± 0.015** | **0.969 ± 0.020** |
| DCNN121     | 2.62 ± 1.66     | 0.950 ± 0.023     | 0.963 ± 0.025     |
| SwinLSTM    | 2.66 ± 1.49     | 0.946 ± 0.020     | 0.960 ± 0.023     |
| IRI 2020    | 6.39 ± 4.53     | 0.720 ± 0.109     | 0.852 ± 0.043     |


Click **Open in Colab** to test without local setup.

---

### Features
- Supports **multichannel spatiotemporal models** for multistep 24-hour input → 24-hour output setup
- Stratified and chronological datasets (Preprocessed GIMs and auxiliary parameters)
- Model registry and configuration system
- Pretrained model download via Hugging Face
- Solar-balanced and storm-aware evaluation experiments
- Tutorials for Colab and local setups.

---

### Framework Status

| Component               | Status      |
|------------------------|-------------|
| HF model & data access | Complete    |
| Model/config registry  | Complete    |
| Evaluation pipeline    | Complete    |
| Visualization tools    | Complete    |
| CLI support            | In Progress |
| Training tutorials     | Planned     |
| Contributor guide      | Planned     |

---

### Local Setup 

```bash
# Clone repository
git clone https://github.com/Mert-chan/IonoBench.git
```
```bash
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
> **Tested on**: Python 3.11.13 · PyTorch 2.5.1 · CUDA 12.4  
> The environment uses `torch==2.5.1`, which requires a compatible CUDA build.  
> PyTorch provides separate wheels for each CUDA version (e.g., `+cu118`, `+cu121`, `+cu124`).  
> Ensure your NVIDIA driver supports the CUDA version used in the installed wheel.
