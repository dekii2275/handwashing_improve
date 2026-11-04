"""
End-to-End Evaluation of WHO Compliance Scoring Index (CSI)
Pillar 3:
1. Loads the best trained Geometric BiLSTM+Attention model
2. Performs windowed inference with Dense Temporal Moving Average Ensembling on each Test video
3. Evaluates Clinical CSI (Ground Truth) vs Predicted CSI (Model)
4. Measures Pearson Correlation (r), Mean Absolute Error (MAE), and Completeness F1
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.train_eval_trajectory import BiLSTM6ClassAttention, extract_18_geom
from src.compliance_engine import ComplianceEngine
from src.features import FeatureStandardizer, impute_missing_landmarks, normalize_spatial


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Dense-Ensemble Video Compliance Evaluation on {device}...", flush=True)

    cache = ROOT / "data" / "landmarks_baseline"
    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())

    # 1. Fit scaler on train split
    train_features = []
    for r in split["train"]:
        stem = r["video_stem"]
        with np.load(cache / (stem + ".npz")) as d:
            raw, presence = d["coords"], d["presence"]
            imputed_raw, _ = impute_missing_landmarks(raw, presence)
            norm_loc = normalize_spatial(imputed_raw, norm_type="norm_loc")
            geom = extract_18_geom(imputed_raw, norm_loc)
            feat = np.concatenate([norm_loc, geom], axis=-1).astype(np.float32)
            train_features.append(feat)

    scaler = FeatureStandardizer().fit(train_features)
    print(f"[*] Fitted FeatureStandardizer on {len(train_features)} train videos.", flush=True)

    # 2. Load trained geometric model
    model = BiLSTM6ClassAttention(input_dim=144, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3).to(device)
    ckpt_path = ROOT / "results" / "who_6class_geometric_144d" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[*] Loaded trained model from {ckpt_path}", flush=True)

    engine = ComplianceEngine()
    window = 48

    gt_csi_list = []
    pred_csi_list = []
    gt_comp_list = []
    pred_comp_list = []
    gt_dur_list = []
    pred_dur_list = []

    video_results = []

    print(f"[*] Running dense inference on {len(split['test'])} Test videos...", flush=True)
    with torch.no_grad():
        for r in split["test"]:
            stem = r["video_stem"]
            with np.load(cache / (stem + ".npz")) as d:
                raw, presence, y_gt = d["coords"], d["presence"], d["labels"]

            # Ground truth compliance
            gt_eval = engine.evaluate(y_gt)

            # Features & Model prediction with updated presence
            imputed_raw, updated_presence = impute_missing_landmarks(raw, presence)
            norm_loc = normalize_spatial(imputed_raw, norm_type="norm_loc")
            geom = extract_18_geom(imputed_raw, norm_loc)
            feat = np.concatenate([norm_loc, geom], axis=-1).astype(np.float32)
            feat_scaled = scaler.transform(feat)

            N = len(feat_scaled)
            prob_accumulator = np.zeros((N, 6), dtype=np.float32)
            count_accumulator = np.zeros(N, dtype=np.int32)

            valid = np.any(updated_presence > 0, axis=1).astype(np.int32)
            counts = np.convolve(valid, np.ones(window, dtype=np.int32), mode="valid")
            starts = np.flatnonzero(counts == window)

            if len(starts) > 0:
                # Batch window inference
                batch_windows = [feat_scaled[s : s + window] for s in starts]
                batch_t = torch.tensor(np.array(batch_windows), dtype=torch.float32, device=device)
                logits = model(batch_t)
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B, 6)

                # Dense temporal accumulation
                for idx, s in enumerate(starts):
                    prob_accumulator[s : s + window] += probs[idx]
                    count_accumulator[s : s + window] += 1

            # Determine frame predictions (1..6) where windows covered, else 0
            pred_labels = np.zeros(N, dtype=np.int32)
            covered = count_accumulator > 0
            if np.any(covered):
                pred_labels[covered] = prob_accumulator[covered].argmax(axis=1) + 1

            pred_eval = engine.evaluate(pred_labels)

            gt_csi = gt_eval["csi_score"]
            pred_csi = pred_eval["csi_score"]
            gt_csi_list.append(gt_csi)
            pred_csi_list.append(pred_csi)
            gt_comp_list.append(gt_eval["completeness_score"])
            pred_comp_list.append(pred_eval["completeness_score"])
            gt_dur_list.append(gt_eval["duration_score"])
            pred_dur_list.append(pred_eval["duration_score"])

            video_results.append({
                "video": stem,
                "gt_csi": gt_csi,
                "pred_csi": pred_csi,
                "csi_error": round(abs(pred_csi - gt_csi), 2),
                "gt_completeness": gt_eval["completeness_score"],
                "pred_completeness": pred_eval["completeness_score"],
                "gt_durations": gt_eval["durations_per_step"],
                "pred_durations": pred_eval["durations_per_step"],
                "missed_steps_gt": gt_eval["missed_steps"],
                "missed_steps_pred": pred_eval["missed_steps"],
            })

    # Summary statistics
    mae_csi = mean_absolute_error(gt_csi_list, pred_csi_list)
    r_csi, p_val = stats.pearsonr(gt_csi_list, pred_csi_list)
    r2_csi = r2_score(gt_csi_list, pred_csi_list)

    mae_comp = mean_absolute_error(gt_comp_list, pred_comp_list)
    r_comp, _ = stats.pearsonr(gt_comp_list, pred_comp_list)

    mae_dur = mean_absolute_error(gt_dur_list, pred_dur_list)
    r_dur, _ = stats.pearsonr(gt_dur_list, pred_dur_list)

    summary = {
        "num_test_videos": len(split["test"]),
        "csi_metrics": {
            "pearson_r": round(float(r_csi), 4),
            "p_value": float(p_val),
            "mae": round(float(mae_csi), 2),
            "r2": round(float(r2_csi), 4),
        },
        "completeness_metrics": {
            "pearson_r": round(float(r_comp), 4),
            "mae": round(float(mae_comp), 2),
        },
        "duration_metrics": {
            "pearson_r": round(float(r_dur), 4),
            "mae": round(float(mae_dur), 2),
        },
        "video_level_results": video_results,
    }

    out_file = ROOT / "results" / "who_compliance_csi_evaluation.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70, flush=True)
    print("WHO COMPLIANCE SCORING INDEX (CSI) - DENSE ENSEMBLE CLINICAL AUDIT", flush=True)
    print("=" * 70, flush=True)
    print(f"Number of Test Videos Evaluated: {len(split['test'])}", flush=True)
    print(f">> CSI Pearson Correlation (r): {r_csi:.4f} (p = {p_val:.2e}) <<", flush=True)
    print(f">> CSI Mean Absolute Error (MAE): {mae_csi:.2f} points (scale 0-100) <<", flush=True)
    print(f">> CSI R² Score: {r2_csi:.4f} <<", flush=True)
    print(f">> Completeness Pearson Correlation (r): {r_comp:.4f} (MAE: {mae_comp:.2f}%) <<", flush=True)
    print(f">> Duration Pearson Correlation (r): {r_dur:.4f} (MAE: {mae_dur:.2f}%) <<", flush=True)
    print("=" * 70, flush=True)
    print(f"Results saved to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
