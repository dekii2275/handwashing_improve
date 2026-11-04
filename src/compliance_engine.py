"""
WHO Compliance Scoring Index (CSI) Engine
Pillar 3: Clinical Compliance Assessment for 6-Step WHO Hand Hygiene Protocol

Evaluates handwashing performance across three dimensions:
1. Completeness Score (S_comp): Has the user completed all 6 necessary WHO steps?
2. Duration Score (S_dur): Did each step and the overall procedure meet clinical duration targets?
3. Order Score (S_order): Did the sequence adhere to the standard WHO sequential progression?

Outputs an aggregated Compliance Scoring Index (CSI) in [0, 100] and actionable clinical feedback.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


WHO_STEP_NAMES = {
    1: "Step 1: Palm to palm",
    2: "Step 2: Palm over dorsum",
    3: "Step 3: Palm to palm interlaced",
    4: "Step 4: Backs of fingers",
    5: "Step 5: Thumb rubbing",
    6: "Step 6: Fingertips rubbing",
}


@dataclass
class StepSegment:
    step_id: int
    step_name: str
    start_frame: int
    end_frame: int
    duration_sec: float


class ComplianceEngine:
    def __init__(
        self,
        fps: float = 30.0,
        min_step_duration_sec: float = 1.0,
        target_step_duration_sec: float = 3.5,
        target_total_duration_sec: float = 20.0,
        weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
    ):
        """
        Args:
            fps: Video frame rate (default 30 fps).
            min_step_duration_sec: Minimum duration for a step to be deemed 'completed'.
            target_step_duration_sec: Recommended clinical duration per step (3.5s per step -> 21s total).
            target_total_duration_sec: Target total duration for 6 WHO steps (20-30 seconds).
            weights: (w_comp, w_dur, w_order) summing to 1.0.
        """
        self.fps = fps
        self.min_step_duration_sec = min_step_duration_sec
        self.target_step_duration_sec = target_step_duration_sec
        self.target_total_duration_sec = target_total_duration_sec
        self.weights = weights
        assert abs(sum(weights) - 1.0) < 1e-4, "Weights must sum to 1.0"

    def filter_predictions(self, raw_preds: np.ndarray, min_run_length: int = 10) -> np.ndarray:
        """
        Run-length filtering to remove high-frequency temporal flickers.
        Runs shorter than min_run_length frames are replaced by surrounding context.
        """
        filtered = raw_preds.copy()
        n = len(filtered)
        if n == 0:
            return filtered

        runs = []
        cur_val = filtered[0]
        cur_start = 0
        for i in range(1, n):
            if filtered[i] != cur_val:
                runs.append((cur_val, cur_start, i))
                cur_val = filtered[i]
                cur_start = i
        runs.append((cur_val, cur_start, n))

        # Filter out short runs
        for idx, (val, start, end) in enumerate(runs):
            length = end - start
            if length < min_run_length:
                prev_val = runs[idx - 1][0] if idx > 0 else (runs[idx + 1][0] if idx + 1 < len(runs) else val)
                filtered[start:end] = prev_val

        return filtered

    def extract_segments(self, labels: np.ndarray) -> List[StepSegment]:
        """Extract continuous temporal action segments from frame-level predictions."""
        n = len(labels)
        if n == 0:
            return []

        segments = []
        cur_lbl = int(labels[0])
        cur_start = 0

        for i in range(1, n):
            lbl = int(labels[i])
            if lbl != cur_lbl:
                if 1 <= cur_lbl <= 6:
                    dur = (i - cur_start) / self.fps
                    segments.append(
                        StepSegment(
                            step_id=cur_lbl,
                            step_name=WHO_STEP_NAMES.get(cur_lbl, f"Step {cur_lbl}"),
                            start_frame=cur_start,
                            end_frame=i,
                            duration_sec=round(dur, 2),
                        )
                    )
                cur_lbl = lbl
                cur_start = i

        if 1 <= cur_lbl <= 6:
            dur = (n - cur_start) / self.fps
            segments.append(
                StepSegment(
                    step_id=cur_lbl,
                    step_name=WHO_STEP_NAMES.get(cur_lbl, f"Step {cur_lbl}"),
                    start_frame=cur_start,
                    end_frame=n,
                    duration_sec=round(dur, 2),
                )
            )

        return segments

    def compute_completeness(self, accumulated_durations: Dict[int, float]) -> Tuple[float, List[int]]:
        """
        Calculates S_comp: fraction of 6 WHO steps performed for >= min_step_duration_sec.
        Returns:
            score: float in [0, 1]
            missed_steps: list of step IDs not completed
        """
        completed = []
        missed = []
        for step in range(1, 7):
            dur = accumulated_durations.get(step, 0.0)
            if dur >= self.min_step_duration_sec:
                completed.append(step)
            else:
                missed.append(step)

        score = len(completed) / 6.0
        return score, missed

    def compute_duration_adequacy(self, accumulated_durations: Dict[int, float]) -> Tuple[float, List[int]]:
        """
        Calculates S_dur: adequacy of time spent on each step and overall procedure.
        Returns:
            score: float in [0, 1]
            suboptimal_steps: steps that met min duration but fell short of clinical target
        """
        step_scores = []
        suboptimal = []
        for step in range(1, 7):
            dur = accumulated_durations.get(step, 0.0)
            ratio = min(1.0, dur / self.target_step_duration_sec)
            step_scores.append(ratio)
            if self.min_step_duration_sec <= dur < self.target_step_duration_sec:
                suboptimal.append(step)

        total_dur = sum(accumulated_durations.values())
        total_ratio = min(1.0, total_dur / self.target_total_duration_sec)

        # 70% per-step duration adequacy + 30% total duration
        score = 0.70 * float(np.mean(step_scores)) + 0.30 * total_ratio
        return float(score), suboptimal

    def compute_order_score(self, segments: List[StepSegment]) -> float:
        """
        Calculates S_order: adherence to the canonical sequence [1, 2, 3, 4, 5, 6].
        Uses normalized Kendall-Tau rank correlation or transition monotonicity.
        """
        if not segments:
            return 0.0

        step_seq = [s.step_id for s in segments]
        # Remove consecutive duplicates in transition sequence
        collapsed = [step_seq[0]]
        for s in step_seq[1:]:
            if s != collapsed[-1]:
                collapsed.append(s)

        if len(collapsed) <= 1:
            return 0.5  # Only 1 step performed

        # Count forward transitions vs backward transitions
        concordant = 0
        total_pairs = 0
        for i in range(len(collapsed)):
            for j in range(i + 1, len(collapsed)):
                total_pairs += 1
                if collapsed[j] > collapsed[i]:
                    concordant += 1

        score = concordant / max(1, total_pairs)
        return float(score)

    def evaluate(self, predictions: np.ndarray) -> Dict:
        """
        Evaluates a single video's prediction sequence.
        Args:
            predictions: 1D array of frame labels (0 to 6, or 1 to 6)
        Returns:
            structured evaluation dict
        """
        filtered = self.filter_predictions(predictions, min_run_length=int(self.fps * 0.4))
        segments = self.extract_segments(filtered)

        # Accumulated duration per step
        accumulated_dur = {s: 0.0 for s in range(1, 7)}
        for seg in segments:
            accumulated_dur[seg.step_id] += seg.duration_sec

        s_comp, missed = self.compute_completeness(accumulated_dur)
        s_dur, suboptimal = self.compute_duration_adequacy(accumulated_dur)
        s_order = self.compute_order_score(segments)

        w_comp, w_dur, w_order = self.weights
        csi = 100.0 * (w_comp * s_comp + w_dur * s_dur + w_order * s_order)

        # Clinical recommendations
        feedback = []
        if missed:
            feedback.append(f"Cảnh báo: Bỏ sót {len(missed)} bước WHO: " + ", ".join([f"Bước {s}" for s in missed]))
        if suboptimal:
            feedback.append(
                f"Lưu ý thời gian: Các bước chưa đủ thời gian khuyến nghị ({self.target_step_duration_sec}s): "
                + ", ".join([f"Bước {s} ({accumulated_dur[s]:.1f}s)" for s in suboptimal])
            )
        total_who_time = sum(accumulated_dur.values())
        if total_who_time < self.target_total_duration_sec:
            feedback.append(f"Tổng thời gian rửa kỹ thuật ({total_who_time:.1f}s) chưa đạt chuẩn tối thiểu ({self.target_total_duration_sec}s).")
        if not feedback:
            feedback.append("Xuất sắc: Quy trình rửa tay đạt chuẩn 6 bước WHO cả về thời lượng và trình tự!")

        return {
            "csi_score": round(csi, 1),
            "completeness_score": round(s_comp * 100, 1),
            "duration_score": round(s_dur * 100, 1),
            "order_score": round(s_order * 100, 1),
            "total_who_duration_sec": round(total_who_time, 1),
            "durations_per_step": {f"Step {s}": round(accumulated_dur[s], 1) for s in range(1, 7)},
            "missed_steps": missed,
            "suboptimal_steps": suboptimal,
            "num_segments": len(segments),
            "clinical_feedback": feedback,
        }
