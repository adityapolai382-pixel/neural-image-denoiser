# Neural Image Denoiser: Deep Autoencoder Benchmark & Deployment

An empirical evaluation and deployment of five autoencoder architectures for restoring noisy handwritten digits from the MNIST dataset under controlled Gaussian corruptions.

---

## Overview

This project benchmarks five distinct autoencoder models on image reconstruction and noise suppression:
- **Undercomplete Autoencoder (UAE):** Strict dimensional bottleneck ($784 \to 256 \to 64 \to 256 \to 784$).
- **Sparse Autoencoder (SAE):** Activity regularization ($L_1 = 10^{-5}$) enforcing sparse latent representations.
- **Denoising Autoencoder (DAE):** Deep symmetric dense architecture with dropout ($p = 0.10$) and hierarchical feature extraction.
- **Convolutional Autoencoder (CAE):** 2D convolutional layers, max-pooling, and upsampling.
- **Contractive Autoencoder (CAE-Contractive):** Custom Keras layer computing the analytic Frobenius norm of the encoder Jacobian to enforce local perturbation invariance.

---

## Benchmark Results

Evaluated on 10,000 unseen test samples corrupted with Additive White Gaussian Noise ($\sigma = 0.30$):

| Model | Parameters | Train Time (s) | PSNR (dB) ↑ | SSIM ↑ | MSE ↓ | Latency (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Denoising AE (Best)** | **1,132,944** | **253.0 s** | **20.82** | **0.8763** | **0.0092** | **0.155** |
| Undercomplete AE | 435,536 | 105.4 s | 20.50 | 0.8409 | 0.0098 | 0.139 |
| Contractive AE | 334,992 | 120.5 s | 19.69 | 0.8036 | 0.0119 | 0.245 |
| Sparse AE | 468,368 | 114.4 s | 11.86 | 0.1188 | 0.0675 | 0.180 |
| Convolutional AE | 12,201 | 1582.7 s | 9.73 | 0.0956 | 0.1140 | 1.368 |

---

## Visualizations

### Model Performance Comparison
Comparative reconstruction performance across PSNR and SSIM:
- Metric comparison (PSNR): `assets/model_comparison_psnr.jpeg`
- Metric comparison (SSIM): `assets/model_comparison_ssim.jpeg`

### Noise and Bottleneck Sweeps
- **Noise factor sweep ($\sigma \in [0.1, 0.5]$):** Analyzed degradation behavior under escalating variance (`assets/noise_sweep_psnr.jpeg`).
- **Latent dimension sweep ($z \in [8, 128]$):** Identified $z = 64$ as the optimal trade-off point between compression and fidelity (`assets/latent_dim_vs_mse.jpeg`).

---

## Interactive Streamlit Web App

The repository includes a standalone Streamlit web application with:
- Image file uploader supporting PNG, JPG, JPEG, and BMP.
- Automatic border-brightness detection to invert bright paper scans to dark MNIST background.
- High-quality anti-aliased Lanczos interpolation to $28 \times 28$.
- Dual-panel visual comparison and one-click PNG download.

### How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app/app.py
```

---

## Repository Structure

```text
├── app/
│   └── app.py                            # Streamlit web app
├── assets/                               # Benchmark plots and visual results
│   ├── sample_noise_comparison.jpeg
│   ├── model_comparison_psnr.jpeg
│   ├── model_comparison_ssim.jpeg
│   └── ...
├── notebooks/
│   └── mnist_autoencoder_denoising.ipynb # Complete experiment workflow
├── report/
│   └── Lab04_Report.tex                  # Formal LaTeX technical report
├── requirements.txt                      # Project dependencies
└── README.md
```
