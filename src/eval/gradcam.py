"""
GhostShot Phase 5 — GradCAM visualization.
Shows which facial regions the model attends to.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_NAMES = {0: "Real", 1: "Fake"}

TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    raw    = img.astype(np.float32) / 255.0
    tensor = TRANSFORM(image=img)["image"].unsqueeze(0)
    return raw, tensor


def run_gradcam(model, img_paths, true_labels, device,
                out_dir=Path("results/figures/gradcam")):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("Installing grad-cam...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "grad-cam"], check=True)
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Target last conv block of EfficientNet-B4
    target_layer = [model.backbone.model.blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)

    n = len(img_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (img_path, true_label) in enumerate(zip(img_paths, true_labels)):
        raw, tensor = load_image(img_path)
        tensor_dev  = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor_dev)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred   = int(probs.argmax())

        target        = [ClassifierOutputTarget(pred)]
        grayscale_cam = cam(input_tensor=tensor_dev, targets=target)[0]
        cam_image     = show_cam_on_image(raw, grayscale_cam, use_rgb=True)

        axes[i, 0].imshow(raw)
        axes[i, 0].set_title(f"True: {CLASS_NAMES[true_label]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(grayscale_cam, cmap="jet")
        axes[i, 1].set_title("GradCAM Heatmap", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cam_image)
        conf = probs[pred] * 100
        color = "green" if pred == true_label else "red"
        axes[i, 2].set_title(
            f"Pred: {CLASS_NAMES[pred]} ({conf:.1f}%)",
            fontsize=9, color=color
        )
        axes[i, 2].axis("off")

    plt.suptitle("GhostShot — GradCAM Explanations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "gradcam_grid.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"✅ GradCAM saved: {out_path}")
