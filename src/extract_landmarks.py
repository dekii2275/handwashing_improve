"""
Hand Landmark Extraction using MediaPipe Hands (Ivanovs & Ivanova Baltic DB&IS 2026)
Extracts 21 3D landmarks for both left and right hands (126 coordinates per frame).
Stores coordinates, presence flags, confidence scores, and frame-level labels.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mediapipe as mp
import numpy as np

from config import (
    COORDS_PER_LANDMARK,
    LANDMARKS_PER_HAND,
    NUM_HANDS,
    RAW_FEATURE_DIM,
)


class HandLandmarkExtractor:
    """MediaPipe Hands landmark extractor for handwashing videos."""

    def __init__(self, min_detection_confidence=0.5, model_path: str | None = None):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "data" / "hand_landmarker.task")

        # Ensure model asset exists
        if not os.path.exists(model_path):
            import urllib.request

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"Downloading hand_landmarker.task to {model_path}...")
            urllib.request.urlretrieve(url, model_path)

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr):
        """
        Process a single BGR video frame.
        Returns:
            coords: np.ndarray shape (126,) [left_hand (63), right_hand (63)]
            presence: np.ndarray shape (2,) [left_present, right_present]
            confidence: np.ndarray shape (2,) [left_conf, right_conf]
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)

        left_coords = np.zeros(
            LANDMARKS_PER_HAND * COORDS_PER_LANDMARK, dtype=np.float32
        )
        right_coords = np.zeros(
            LANDMARKS_PER_HAND * COORDS_PER_LANDMARK, dtype=np.float32
        )
        presence = np.array([0, 0], dtype=np.int32)
        confidence = np.array([0.0, 0.0], dtype=np.float32)

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(
                results.hand_landmarks, results.handedness
            ):
                label = handedness[0].category_name  # 'Left' or 'Right'
                score = handedness[0].score

                flat_lm = []
                for lm in hand_landmarks:
                    flat_lm.extend([lm.x, lm.y, lm.z])
                flat_lm = np.array(flat_lm, dtype=np.float32)

                if label == "Left":
                    left_coords = flat_lm
                    presence[0] = 1
                    confidence[0] = score
                else:
                    right_coords = flat_lm
                    presence[1] = 1
                    confidence[1] = score

        coords = np.concatenate([left_coords, right_coords])
        return coords, presence, confidence

    def extract_from_video(
        self,
        video_path: Path,
        annotation_path: Path | None = None,
        output_path: Path | None = None,
    ):
        """
        Extract landmark sequences from an entire video.
        Synchronizes with ground-truth frame labels from annotation JSON.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Load frame-level labels if annotation exists
        labels = None
        if annotation_path and annotation_path.exists():
            with open(annotation_path, "r", encoding="utf-8") as f:
                annot = json.load(f)
                labels = [item["code"] for item in annot.get("labels", [])]

        all_coords = []
        all_presence = []
        all_confidence = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            coords, presence, confidence = self.process_frame(frame)
            all_coords.append(coords)
            all_presence.append(presence)
            all_confidence.append(confidence)
            frame_idx += 1

        cap.release()

        all_coords = np.array(all_coords, dtype=np.float32)
        all_presence = np.array(all_presence, dtype=np.int32)
        all_confidence = np.array(all_confidence, dtype=np.float32)

        # Truncate or align labels with frame count
        if labels is not None:
            n = min(len(labels), len(all_coords))
            all_coords = all_coords[:n]
            all_presence = all_presence[:n]
            all_confidence = all_confidence[:n]
            final_labels = np.array(labels[:n], dtype=np.int64)
        else:
            final_labels = np.zeros(len(all_coords), dtype=np.int64)

        result = {
            "coords": all_coords,  # (N, 126)
            "presence": all_presence,  # (N, 2)
            "confidence": all_confidence,  # (N, 2)
            "labels": final_labels,  # (N,)
            "total_frames": len(all_coords),
            "fps": fps,
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                coords=all_coords,
                presence=all_presence,
                confidence=all_confidence,
                labels=final_labels,
                fps=fps,
            )

        return result

    def close(self):
        if hasattr(self, "detector"):
            self.detector.close()


def generate_synthetic_landmarks_for_annotation(
    annotation_path: Path, output_path: Path, seed: int = 42
):
    """
    Generate anatomically plausible hand landmarks conditioned on the real ground-truth JSON labels.
    Useful for pipeline verification, benchmarking, and unit testing when raw video download is deferred.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        annot = json.load(f)
        labels = [item["code"] for item in annot.get("labels", [])]

    n_frames = len(labels)
    rng = np.random.RandomState(seed + n_frames)

    # Base template of a hand (21 landmarks) centered at (0.5, 0.5, 0.0)
    # Approximate relative finger joints
    template_left = rng.randn(LANDMARKS_PER_HAND, 3) * 0.05
    template_left[:, 0] += 0.4
    template_left[:, 1] += 0.5

    template_right = rng.randn(LANDMARKS_PER_HAND, 3) * 0.05
    template_right[:, 0] += 0.6
    template_right[:, 1] += 0.5

    all_coords = np.zeros((n_frames, RAW_FEATURE_DIM), dtype=np.float32)
    presence = np.ones((n_frames, 2), dtype=np.int32)
    confidence = np.full((n_frames, 2), 0.9, dtype=np.float32)

    # Class-specific motion frequencies and amplitudes to reflect different WHO gestures
    motion_params = {
        0: (0.5, 0.02),  # Other: slow irregular
        1: (2.0, 0.04),  # Palm to palm: back-and-forth x-motion
        2: (1.5, 0.035),  # Palm over dorsum
        3: (1.8, 0.03),  # Fingers interlaced
        4: (1.2, 0.04),  # Backs of fingers
        5: (2.5, 0.05),  # Rotational thumb
        6: (2.2, 0.045),  # Fingertips to palm
    }

    t = np.arange(n_frames) / 16.0  # 16 FPS
    for i in range(n_frames):
        c = labels[i]
        freq, amp = motion_params.get(c, (1.0, 0.03))
        # Add smooth periodic motion + slight noise
        offset_l = np.sin(2 * np.pi * freq * t[i]) * amp + rng.randn() * 0.005
        offset_r = -np.sin(2 * np.pi * freq * t[i]) * amp + rng.randn() * 0.005

        cur_left = template_left.copy()
        cur_left[:, 0] += offset_l
        cur_left[:, 1] += offset_l * 0.5

        cur_right = template_right.copy()
        cur_right[:, 0] += offset_r
        cur_right[:, 1] += -offset_r * 0.5

        # Occasional missing hand detection (simulate soap occlusion) ~ 5% of frames
        if rng.rand() < 0.05:
            if rng.rand() < 0.5:
                cur_left.fill(0)
                presence[i, 0] = 0
                confidence[i, 0] = 0.0
            else:
                cur_right.fill(0)
                presence[i, 1] = 0
                confidence[i, 1] = 0.0

        all_coords[i, :63] = cur_left.flatten()
        all_coords[i, 63:] = cur_right.flatten()

    labels_arr = np.array(labels, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coords=all_coords,
        presence=presence,
        confidence=confidence,
        labels=labels_arr,
        fps=16.0,
    )
    return n_frames
