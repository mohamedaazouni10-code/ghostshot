"""
GhostShot — Colab environment checker.
Run this cell first in every Colab session to verify the stack is healthy.
"""
import sys
import importlib
import torch


REQUIRED = [
    "torch", "torchvision", "cv2", "timm",
    "albumentations", "wandb", "sklearn",
    "deepface", "facenet_pytorch", "tqdm",
]

def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "n/a")
            print(f"  ✓  {pkg:<22} {version}")
        except ImportError:
            print(f"  ✗  {pkg:<22} NOT FOUND")
            all_ok = False
    return all_ok


def check_gpu() -> bool:
    print("\n── GPU ──────────────────────────────────────────────────")
    if not torch.cuda.is_available():
        print("  ✗  No CUDA GPU detected.")
        print("     → Runtime > Change runtime type > T4 GPU")
        return False

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1024**3
        print(f"  ✓  GPU {i}: {props.name} | {vram:.1f} GB VRAM")

    t = torch.zeros(1, device="cuda")
    del t
    torch.cuda.empty_cache()
    print(f"  ✓  CUDA version : {torch.version.cuda}")
    print(f"  ✓  cuDNN version: {torch.backends.cudnn.version()}")
    return True


def check_backbone() -> bool:
    print("\n── Model backbone ───────────────────────────────────────")
    try:
        import timm
        model = timm.create_model("efficientnet_b4", pretrained=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓  EfficientNet-B4 loaded ({params:.1f}M parameters)")
        return True
    except Exception as e:
        print(f"  ✗  timm backbone failed: {e}")
        return False


def check_face_stack() -> bool:
    print("\n── Face stack ───────────────────────────────────────────")
    results = {}

    try:
        from facenet_pytorch import MTCNN
        MTCNN(keep_all=False, device="cpu")
        print("  ✓  MTCNN (facenet-pytorch)")
        results["mtcnn"] = True
    except Exception as e:
        print(f"  ✗  MTCNN: {e}")
        results["mtcnn"] = False

    try:
        from deepface import DeepFace
        print("  ✓  DeepFace")
        results["deepface"] = True
    except Exception as e:
        print(f"  ✗  DeepFace: {e}")
        results["deepface"] = False

    return all(results.values())


def run_all_checks() -> None:
    print("=" * 56)
    print("  GhostShot — Environment Check")
    print(f"  Python {sys.version.split()[0]}")
    print(f"  PyTorch {torch.__version__}")
    print("=" * 56)

    print("\n── Packages ─────────────────────────────────────────────")
    pkg_ok  = check_packages()
    gpu_ok  = check_gpu()
    bb_ok   = check_backbone()
    face_ok = check_face_stack()

    print("\n── Summary ──────────────────────────────────────────────")
    status = {
        "Packages"  : pkg_ok,
        "GPU"       : gpu_ok,
        "Backbone"  : bb_ok,
        "Face stack": face_ok,
    }
    all_passed = True
    for name, ok in status.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")
        if not ok:
            all_passed = False

    print("=" * 56)
    if all_passed:
        print("  All checks passed — ready to build GhostShot!")
    else:
        print("  Fix the items above before proceeding.")
    print("=" * 56)


if __name__ == "__main__":
    run_all_checks()
