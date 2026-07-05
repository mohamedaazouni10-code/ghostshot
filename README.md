# GhostShot — Deepfake Detection as a Cybersecurity Attack

> Framing deepfake identity images as an **active cyber attack** against
> face-based authentication systems — with an end-to-end pipeline.

## Results
- **AUC = 0.84** on test set
- **87% recall** on deepfakes
- **25% Attack Success Rate Reduction (ASRR)**

## Project Structure
ghostshot/
├── configs/         ← YAML config
├── notebooks/       ← Colab notebooks (Phase 1-5 + demo)
├── src/
│   ├── data/        ← dataset & face-crop pipeline
│   ├── models/      ← EfficientNet-B4 + FFT branch
│   ├── attack/      ← attack simulation & ArcFace API
│   ├── eval/        ← metrics, GradCAM, ASRR
│   └── utils/       ← seed, env check
└── requirements.txt

## Notebooks
| Notebook | Description |
|----------|-------------|
| 00_setup.ipynb | Environment setup |
| 01_data_pipeline.ipynb | Dataset & face crops |
| 02_attack_simulation.ipynb | Attack simulation |
| 03_train_detector.ipynb | Model training |
| 04_evaluation.ipynb | Evaluation & GradCAM |
| demo_ghostshot.ipynb | Live demo |

## Quick Start
```bash
git clone https://github.com/mohamedaazouni10-code/ghostshot
cd ghostshot
pip install -r requirements.txt
```

## Dataset
Celeb-DF v2 — 95,216 face crops (23K real + 72K fake)

## Model
EfficientNet-B4 + FFT branch — 19.3M parameters

## Author
Mohamed Aazouni — PFE 2024/2025
EOF
