"""
GhostShot Phase 3 — Mock face-authentication API.
Simulates a real face-auth system using ArcFace embeddings.
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from deepface import DeepFace
from tqdm import tqdm


MODEL_NAME = "ArcFace"
DETECTOR   = "retinaface"
THRESHOLD  = 0.68
EMBED_DIM  = 512


class FaceAuthAPI:
    def __init__(
        self,
        db_path: Path = Path("data/attack_samples/face_auth_db.pkl"),
        threshold: float = THRESHOLD,
    ):
        self.db_path   = Path(db_path)
        self.threshold = threshold
        self.enrolled  = self._load_db()

    def _load_db(self) -> dict:
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                db = pickle.load(f)
            print(f"[FaceAuthAPI] Loaded {len(db)} enrolled identities.")
            return db
        return {}

    def _save_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.enrolled, f)

    def _get_embedding(self, img_path: str) -> Optional[np.ndarray]:
        try:
            result = DeepFace.represent(
                img_path          = img_path,
                model_name        = MODEL_NAME,
                detector_backend  = DETECTOR,
                enforce_detection = True,
            )
            return np.array(result[0]["embedding"])
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def enroll(self, identity_id: str, img_path: str) -> bool:
        emb = self._get_embedding(img_path)
        if emb is None:
            print(f"  [enroll] No face detected: {img_path}")
            return False
        self.enrolled[identity_id] = {"embedding": emb, "img_path": img_path}
        self._save_db()
        return True

    def verify(self, identity_id: str, probe_path: str) -> dict:
        if identity_id not in self.enrolled:
            raise ValueError(f"Identity '{identity_id}' not enrolled.")
        probe_emb = self._get_embedding(probe_path)
        if probe_emb is None:
            return {
                "identity_id": identity_id,
                "probe_path":  probe_path,
                "similarity":  0.0,
                "passed":      False,
                "threshold":   self.threshold,
                "error":       "no_face_detected",
            }
        enrolled_emb = self.enrolled[identity_id]["embedding"]
        similarity   = self._cosine_similarity(enrolled_emb, probe_emb)
        return {
            "identity_id": identity_id,
            "probe_path":  probe_path,
            "similarity":  round(similarity, 4),
            "passed":      similarity >= self.threshold,
            "threshold":   self.threshold,
        }

    def bulk_enroll(self, identity_img_map: dict) -> int:
        success = 0
        for identity_id, img_path in tqdm(
            identity_img_map.items(), desc="Enrolling victims"
        ):
            if self.enroll(identity_id, img_path):
                success += 1
        print(f"[bulk_enroll] {success}/{len(identity_img_map)} enrolled.")
        return success

    def list_enrolled(self) -> list:
        return list(self.enrolled.keys())
