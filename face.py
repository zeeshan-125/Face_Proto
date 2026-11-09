"""Video frame analysis for the face verification prototype (Step B).

MediaPipe's Face Landmarker task model must be supplied separately. Set
``FACE_LANDMARKER_MODEL`` to the path of a compatible ``.task`` model file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ANALYSIS_WIDTH = 640
EYES_OPEN_MAX = 0.30
MIN_FACE_WIDTH_PX = 100
MIN_SHARPNESS = 40.0
MIN_BRIGHTNESS = 40.0
SAME_PERSON_THRESHOLD = 0.363
ENROLLED_MATCH_THRESHOLD = 0.363


class VideoReadError(Exception):
    """Raised when OpenCV cannot open or decode a video."""


class ModelConfigurationError(Exception):
    """Raised when the MediaPipe Face Landmarker model is not configured."""


class FaceModelsConfigurationError(Exception):
    """Raised when YuNet or SFace model paths are not configured."""


class FaceprintQualityError(Exception):
    """Raised when selected frames cannot produce enough faceprints."""


@dataclass
class FrameSample:
    frame_index: int
    timestamp_seconds: float
    face_count: int
    eye_closed: float | None
    smiling: float | None


@dataclass
class SelectedFrame:
    """The sharpest eligible resized frame for one second of the clip."""

    second: int
    frame_index: int
    sharpness: float
    face_width_px: float
    face_height_px: float
    brightness: float
    image_bgr: np.ndarray


@dataclass
class VideoAnalysis:
    fps: float
    frames: list[FrameSample]
    selected_frames: list[SelectedFrame]

    def summary(self) -> dict:
        return {
            "fps": self.fps,
            "frame_count": len(self.frames),
            "face_counts": [frame.face_count for frame in self.frames],
            "eye_closed_scores": [frame.eye_closed for frame in self.frames],
            "smile_scores": [frame.smiling for frame in self.frames],
            "selected_frames": [
                {
                    "second": frame.second,
                    "frame_index": frame.frame_index,
                    "sharpness": frame.sharpness,
                    "face_width_px": frame.face_width_px,
                    "face_height_px": frame.face_height_px,
                    "brightness": frame.brightness,
                }
                for frame in self.selected_frames
            ],
        }


@dataclass
class FaceprintResult:
    embeddings: list[np.ndarray]
    average: np.ndarray
    pairwise_similarities: list[float]
    enrolled_similarity: float | None
    same_person: bool
    enrolled_match: bool | None

    def summary(self, *, include_faceprint: bool = False) -> dict:
        result = {
            "photo_count": len(self.embeddings),
            "pairwise_similarities": self.pairwise_similarities,
            "enrolled_similarity": self.enrolled_similarity,
            "same_person": self.same_person,
            "enrolled_match": self.enrolled_match,
        }
        if include_faceprint:
            result["faceprint"] = [float(value) for value in self.average]
        return result


@dataclass
class ChallengeEvent:
    action: str
    timestamp_seconds: float
    frame_index: int

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_index": self.frame_index,
        }


def detect_events(
    frames: list[FrameSample],
    *,
    blink_closed_threshold: float = 0.5,
    blink_open_threshold: float = 0.3,
    smile_threshold: float = 0.6,
    smile_min_frames: int = 3,
) -> list[ChallengeEvent]:
    """Detect debounced blinks and sustained smiles, ordered by detection time."""
    events: list[ChallengeEvent] = []
    blink_is_closed = False
    smile_run: list[FrameSample] = []
    smile_emitted = False

    for frame in frames:
        # Ignore eye state when the face is not uniquely visible. Keep a partial
        # blink armed across those frames, but do not count an absence as an open eye.
        if frame.face_count == 1 and frame.eye_closed is not None:
            if not blink_is_closed and frame.eye_closed > blink_closed_threshold:
                blink_is_closed = True
            elif blink_is_closed and frame.eye_closed < blink_open_threshold:
                blink_is_closed = False
                events.append(ChallengeEvent("blink", frame.timestamp_seconds, frame.frame_index))

        # A missing/ambiguous face breaks the run, since smile frames must be
        # consecutive valid single-face observations.
        if frame.face_count != 1 or frame.smiling is None or frame.smiling <= smile_threshold:
            smile_run = []
            smile_emitted = False
            continue

        smile_run.append(frame)
        if len(smile_run) >= smile_min_frames and not smile_emitted:
            qualifying_frame = smile_run[smile_min_frames - 1]
            events.append(
                ChallengeEvent("smile", qualifying_frame.timestamp_seconds, qualifying_frame.frame_index)
            )
            smile_emitted = True

    return sorted(events, key=lambda event: (event.timestamp_seconds, event.frame_index))


def challenge_completed(actions: list[str], events: list[ChallengeEvent]) -> bool:
    """Return whether requested actions occur as an ordered subsequence of events."""
    next_action = 0
    for event in events:
        if next_action < len(actions) and event.action == actions[next_action]:
            next_action += 1
            if next_action == len(actions):
                return True
    return not actions


def _blendshape_score(categories, name: str) -> float:
    for category in categories:
        if category.category_name == name:
            return float(category.score)
    return 0.0


def _scores(result) -> tuple[float, float]:
    """Return normalized eye-closed and smile scores from blendshapes."""
    if not result.face_blendshapes:
        return 0.0, 0.0
    categories = result.face_blendshapes[0]
    closed = (
        _blendshape_score(categories, "eyeBlinkLeft")
        + _blendshape_score(categories, "eyeBlinkRight")
    ) / 2
    smile = (
        _blendshape_score(categories, "mouthSmileLeft")
        + _blendshape_score(categories, "mouthSmileRight")
    ) / 2
    return closed, smile


def analyze_video(video_path: str | os.PathLike[str], model_path: str | None = None) -> VideoAnalysis:
    """Analyze every decoded frame and retain the sharpest open-eye single-face frame per second."""
    model_path = model_path or os.getenv("FACE_LANDMARKER_MODEL")
    if not model_path or not Path(model_path).is_file():
        raise ModelConfigurationError(
            "Set FACE_LANDMARKER_MODEL to a downloaded MediaPipe Face Landmarker .task model."
        )

    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError("Could not open video")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.fspath(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
        output_face_blendshapes=True,
    )
    samples: list[FrameSample] = []
    selected_by_second: dict[int, SelectedFrame] = {}

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                height, width = frame.shape[:2]
                if width > ANALYSIS_WIDTH:
                    new_height = max(1, round(height * ANALYSIS_WIDTH / width))
                    frame = cv2.resize(frame, (ANALYSIS_WIDTH, new_height), interpolation=cv2.INTER_AREA)

                timestamp = frame_index / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                face_count = len(result.face_landmarks)
                eye_closed, smiling = _scores(result) if face_count == 1 else (None, None)
                samples.append(
                    FrameSample(frame_index, timestamp, face_count, eye_closed, smiling)
                )

                if face_count == 1 and eye_closed < EYES_OPEN_MAX:
                    second = int(timestamp)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    landmarks = result.face_landmarks[0]
                    xs = [point.x for point in landmarks]
                    ys = [point.y for point in landmarks]
                    face_width = (max(xs) - min(xs)) * frame.shape[1]
                    face_height = (max(ys) - min(ys)) * frame.shape[0]
                    brightness = float(gray.mean())
                    current = selected_by_second.get(second)
                    if current is None or sharpness > current.sharpness:
                        selected_by_second[second] = SelectedFrame(
                            second,
                            frame_index,
                            sharpness,
                            face_width,
                            face_height,
                            brightness,
                            frame.copy(),
                        )
                frame_index += 1
    except OSError as exc:
        raise ModelConfigurationError(
            "MediaPipe could not load its native runtime. On this Windows machine, Application Control "
            "blocked the runtime (WinError 4551). Ask your administrator to allow the MediaPipe runtime, "
            "or run this prototype on a machine where it is permitted."
        ) from exc
    finally:
        capture.release()

    if not samples:
        raise VideoReadError("Video contains no decodable frames")
    return VideoAnalysis(fps, samples, [selected_by_second[key] for key in sorted(selected_by_second)])


def first_failure(analysis: VideoAnalysis, actions: list[str]) -> str | None:
    """Apply Step D checks in order and return the first failure reason."""
    total = len(analysis.frames)
    if total == 0:
        return "bad_video"

    no_face_ratio = sum(frame.face_count == 0 for frame in analysis.frames) / total
    if no_face_ratio > 0.30:
        return "no_face"

    multiple_faces_ratio = sum(frame.face_count >= 2 for frame in analysis.frames) / total
    if multiple_faces_ratio > 0.10:
        return "multiple_faces"

    events = detect_events(analysis.frames)
    if not challenge_completed(actions, events):
        return "challenge_not_done"

    quality_photos = len(good_selected_frames(analysis.selected_frames))
    if quality_photos < 3:
        return "low_quality"

    # Step E appends same-person and enrolled-person checks after quality.
    return None


def good_selected_frames(selected_frames: list[SelectedFrame]) -> list[SelectedFrame]:
    """Return selected photos meeting the Step D size, blur, and brightness checks."""
    return [
        selected
        for selected in selected_frames
        if selected.face_width_px >= MIN_FACE_WIDTH_PX
        and selected.sharpness >= MIN_SHARPNESS
        and selected.brightness >= MIN_BRIGHTNESS
    ]


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def build_faceprints(
    selected_frames: list[SelectedFrame],
    enrolled_faceprint: list[float] | None = None,
    *,
    yunet_model: str | None = None,
    sface_model: str | None = None,
) -> FaceprintResult:
    """Align selected faces with YuNet, create SFace embeddings, and compare them."""
    yunet_model = yunet_model or os.getenv("FACE_YUNET_MODEL")
    sface_model = sface_model or os.getenv("FACE_SFACE_MODEL")
    if (
        not yunet_model
        or not sface_model
        or not Path(yunet_model).is_file()
        or not Path(sface_model).is_file()
    ):
        raise FaceModelsConfigurationError(
            "Set FACE_YUNET_MODEL and FACE_SFACE_MODEL to the OpenCV Zoo ONNX model paths."
        )

    try:
        detector = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320), 0.8, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(sface_model, "")
    except cv2.error as exc:
        raise FaceModelsConfigurationError("Could not load the YuNet and SFace ONNX models") from exc
    embeddings: list[np.ndarray] = []

    for selected in selected_frames:
        frame = selected.image_bgr
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, detections = detector.detect(frame)
        if detections is None or len(detections) == 0:
            raise FaceprintQualityError("YuNet could not find a face in a selected photo")

        # The video stage already required exactly one MediaPipe face. If YuNet
        # reports more than one, use its highest confidence detection.
        detection = max(detections, key=lambda row: float(row[-1]))
        aligned = recognizer.alignCrop(frame, detection.reshape(1, -1))
        embedding = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        if embedding.size != 128 or not np.all(np.isfinite(embedding)) or np.linalg.norm(embedding) == 0:
            raise FaceModelsConfigurationError("SFace did not return a finite 128-number faceprint")
        embeddings.append(embedding)

    if len(embeddings) < 3:
        raise FaceprintQualityError("Fewer than three selected photos produced a faceprint")

    pairwise: list[float] = []
    same_person = True
    for first_index in range(len(embeddings)):
        for second_index in range(first_index + 1, len(embeddings)):
            similarity = _cosine_similarity(embeddings[first_index], embeddings[second_index])
            pairwise.append(similarity)
            if similarity < SAME_PERSON_THRESHOLD:
                same_person = False

    average = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    enrolled_similarity = (
        _cosine_similarity(average, np.asarray(enrolled_faceprint, dtype=np.float32))
        if enrolled_faceprint is not None
        else None
    )
    return FaceprintResult(
        embeddings=embeddings,
        average=average,
        pairwise_similarities=pairwise,
        enrolled_similarity=enrolled_similarity,
        same_person=same_person,
        enrolled_match=(enrolled_similarity >= ENROLLED_MATCH_THRESHOLD)
        if enrolled_similarity is not None
        else None,
    )
