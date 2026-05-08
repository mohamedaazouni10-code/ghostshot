"""
GhostShot Phase 3 — Attack probe runner.
Generates deepfakes and probes the face-auth API.
"""
import csv
from pathlib import Path

from tqdm import tqdm

from .face_auth_api import FaceAuthAPI
from .generator import DeepfakeGenerator


LOG_PATH = Path("data/attack_samples/attack_log.csv")
LOG_FIELDS = [
    "victim_id",
    "source_id",
    "probe_path",
    "generator",
    "similarity",
    "passed",
    "threshold",
    "error",
]


def run_attack_campaign(
    api: FaceAuthAPI,
    generator: DeepfakeGenerator,
    victim_source_pairs: list,
    n_probes_per_victim: int = 10,
    log_path: Path = LOG_PATH,
) -> list:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for victim_id, victim_img, source_id, source_img in tqdm(
            victim_source_pairs, desc="Attack campaign"
        ):
            for probe_idx in range(n_probes_per_victim):
                probe_path = generator.generate(
                    source_path = source_img,
                    target_path = victim_img,
                    identity_id = victim_id,
                    probe_idx   = probe_idx,
                )

                if probe_path is None:
                    row = {
                        "victim_id":  victim_id,
                        "source_id":  source_id,
                        "probe_path": "",
                        "generator":  generator.method,
                        "similarity": 0.0,
                        "passed":     False,
                        "threshold":  api.threshold,
                        "error":      "generation_failed",
                    }
                    writer.writerow(row)
                    all_results.append(row)
                    continue

                result = api.verify(victim_id, probe_path)

                row = {
                    "victim_id":  victim_id,
                    "source_id":  source_id,
                    "probe_path": probe_path,
                    "generator":  generator.method,
                    "similarity": result["similarity"],
                    "passed":     result["passed"],
                    "threshold":  result["threshold"],
                    "error":      result.get("error", ""),
                }
                writer.writerow(row)
                f.flush()
                all_results.append(row)

    print(f"\n[probe] Campaign complete. Log: {log_path}")
    _print_summary(all_results)
    return all_results


def _print_summary(results: list) -> None:
    total   = len(results)
    success = sum(1 for r in results if r["passed"])
    failed  = total - success
    asr     = success / total * 100 if total else 0

    print("\n── Attack Campaign Summary ───────────────────────────")
    print(f"  Total probes   : {total}")
    print(f"  Passed (fooled): {success}  ({asr:.1f}%)")
    print(f"  Blocked        : {failed}")
    print(f"  Attack Success Rate (ASR): {asr:.1f}%")
    print("──────────────────────────────────────────────────────")
