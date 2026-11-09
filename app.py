"""Local onboarding and offboarding face verification API."""

from __future__ import annotations

import json
import math
import os
import tempfile
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from profile_store import create_profile, get_profile, normalize_profile_id

app = FastAPI(title="Face verification prototype")
MAX_VIDEO_BYTES = 25 * 1024 * 1024
EXPECTED_ACTIONS = {"blink", "smile"}
UI_PATH = Path(__file__).parent / "static" / "index.html"


def _load_project_env() -> None:
    """Load local keys and model paths before OpenCV initializes its inference engine."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        return
    allowed_keys = {
        "FACE_API_KEY",
        "FACE_LANDMARKER_MODEL",
        "FACE_YUNET_MODEL",
        "FACE_SFACE_MODEL",
        "OPENCV_FORCE_DNN_ENGINE",
    }
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        key = key.strip()
        if not separator or key not in allowed_keys:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.endswith("_MODEL") and value:
            model_path = Path(value)
            if not model_path.is_absolute():
                value = str(Path(__file__).parent / model_path)
        os.environ.setdefault(key, value)


_load_project_env()

from face import (  # noqa: E402
    FaceModelsConfigurationError,
    FaceprintQualityError,
    ModelConfigurationError,
    VideoReadError,
    analyze_video,
    build_faceprints,
    detect_events,
    first_failure,
    good_selected_frames,
)


def _error(status: int, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail, **extra})


def _ui_is_authorized(request: Request) -> JSONResponse | None:
    """Require a configured server key and a local same-origin browser request."""
    if not os.getenv("FACE_API_KEY"):
        return _error(503, "FACE_API_KEY is not configured in the server environment or project .env file")
    origin = request.headers.get("origin")
    host = request.headers.get("host", "")
    host_name = host.lower().split(":", 1)[0]
    expected_origin = f"{request.url.scheme}://{host}".rstrip("/")
    if (
        host_name not in {"127.0.0.1", "localhost"}
        or not origin
        or origin.rstrip("/") != expected_origin
    ):
        return _error(403, "This endpoint accepts requests from the local UI origin only")
    return None


async def _read_form(request: Request):
    try:
        return await request.form(), None
    except Exception:
        return None, _error(422, "Expected multipart form data")


def _parse_json_field(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be JSON")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc


def _valid_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _parse_actions(form) -> list[str] | JSONResponse:
    try:
        actions = _parse_json_field(form.get("actions"), "actions")
    except ValueError as exc:
        return _error(422, str(exc))
    if (
        not isinstance(actions, list)
        or not actions
        or any(not isinstance(action, str) or action not in EXPECTED_ACTIONS for action in actions)
    ):
        return _error(422, 'actions must be a non-empty list containing only "blink" and "smile"')
    return actions


def _parse_faceprint(value: Any) -> list[float] | JSONResponse:
    try:
        faceprint = _parse_json_field(value, "face_data")
    except ValueError as exc:
        return _error(422, str(exc))
    if not isinstance(faceprint, list) or len(faceprint) != 128 or any(not _valid_number(v) for v in faceprint):
        return _error(422, "face_data must contain exactly 128 finite numbers")
    return faceprint


def _result(status_code: int, content: dict) -> tuple[int, dict, Any | None]:
    return status_code, content, None


async def _process_capture(form, enrolled_faceprint: list[float] | None) -> tuple[int, dict, Any | None]:
    """Run the common liveness, quality, and faceprint steps for either workflow."""
    actions = _parse_actions(form)
    if isinstance(actions, JSONResponse):
        return actions.status_code, json.loads(actions.body), None

    video = form.get("video")
    if video is None or not hasattr(video, "read"):
        return 422, {"detail": "video file is required"}, None

    temp_path: str | None = None
    try:
        video_bytes = await video.read(MAX_VIDEO_BYTES + 1)
        if len(video_bytes) > MAX_VIDEO_BYTES:
            return 413, {"detail": "Video exceeds the 25 MB limit"}, None
        if not video_bytes:
            return 422, {"reason": "bad_video", "detail": "Video file is empty"}, None

        suffix = Path(getattr(video, "filename", "") or "").suffix
        # On Windows, OpenCV cannot open a NamedTemporaryFile while Python still
        # holds its exclusive handle. Close the file before handing the path to
        # VideoCapture, then remove it in the outer finally block.
        with tempfile.NamedTemporaryFile(prefix="face-verify-", suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            temp.write(video_bytes)
            temp.flush()
        try:
            analysis = analyze_video(temp_path)
        except ModelConfigurationError as exc:
            return 503, {"detail": str(exc)}, None
        except VideoReadError as exc:
            return 422, {"reason": "bad_video", "detail": str(exc)}, None

        events = detect_events(analysis.frames)
        failure = first_failure(analysis, actions)
        if failure:
            return 200, {
                "status": "rejected",
                "reason": failure,
                "analysis": analysis.summary(),
                "detected_events": [event.as_dict() for event in events],
            }, None

        try:
            faceprints = build_faceprints(
                good_selected_frames(analysis.selected_frames),
                enrolled_faceprint,
            )
        except FaceModelsConfigurationError as exc:
            return 503, {"detail": str(exc)}, None
        except FaceprintQualityError:
            return 200, {
                "status": "rejected",
                "reason": "low_quality",
                "analysis": analysis.summary(),
                "detected_events": [event.as_dict() for event in events],
            }, None

        base = {
            "analysis": analysis.summary(),
            "detected_events": [event.as_dict() for event in events],
            "challenge_passed": True,
            "faceprints": faceprints.summary(),
        }
        if not faceprints.same_person:
            return 200, {"status": "rejected", "reason": "not_same_person", **base}, faceprints
        if enrolled_faceprint is not None and not faceprints.enrolled_match:
            return 200, {"status": "rejected", "reason": "not_matched", **base}, faceprints

        status = "verified" if enrolled_faceprint is not None else "enrollment_ready"
        return 200, {"status": status, **base}, faceprints
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass
        await video.close()


async def _authenticated_ui_form(request: Request):
    auth_error = _ui_is_authorized(request)
    if auth_error:
        return None, auth_error
    return await _read_form(request)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(UI_PATH)


@app.post("/ui/enroll", include_in_schema=False)
async def enroll_from_ui(request: Request) -> JSONResponse:
    form, error = await _authenticated_ui_form(request)
    if error:
        return error
    try:
        raw_profile_id = form.get("profile_id")
        profile_id = normalize_profile_id(raw_profile_id if isinstance(raw_profile_id, str) else "")
    except ValueError as exc:
        return _error(422, str(exc))
    if get_profile(profile_id):
        return _error(409, "This Profile ID is already enrolled. Use a different ID for a new onboarding capture.", reason="already_enrolled")

    status_code, content, faceprints = await _process_capture(form, None)
    if status_code != 200 or content.get("status") != "enrollment_ready":
        return JSONResponse(status_code=status_code, content=content)
    try:
        saved = create_profile(profile_id, [float(value) for value in faceprints.average])
    except sqlite3.IntegrityError:
        return _error(409, "This Profile ID was enrolled by another request. Use a different ID.", reason="already_enrolled")
    content["status"] = "enrolled"
    content["profile"] = saved
    return JSONResponse(status_code=200, content=content)


@app.post("/ui/offboard", include_in_schema=False)
async def offboard_from_ui(request: Request) -> JSONResponse:
    form, error = await _authenticated_ui_form(request)
    if error:
        return error
    try:
        raw_profile_id = form.get("profile_id")
        profile_id = normalize_profile_id(raw_profile_id if isinstance(raw_profile_id, str) else "")
    except ValueError as exc:
        return _error(422, str(exc))
    enrolled = get_profile(profile_id)
    if not enrolled:
        return _error(404, "No onboarding faceprint was found for this Profile ID. Complete onboarding first.", reason="profile_not_found")

    status_code, content, _ = await _process_capture(form, enrolled["faceprint"])
    content["profile"] = {"profile_id": profile_id}
    return JSONResponse(status_code=status_code, content=content)


@app.post("/ui/verify", include_in_schema=False)
async def verify_from_ui(request: Request) -> JSONResponse:
    """Compatibility route for the previous UI version that submitted face_data directly."""
    form, error = await _authenticated_ui_form(request)
    if error:
        return error
    faceprint = _parse_faceprint(form.get("face_data"))
    if isinstance(faceprint, JSONResponse):
        return faceprint
    status_code, content, _ = await _process_capture(form, faceprint)
    return JSONResponse(status_code=status_code, content=content)


@app.post("/verify")
async def verify(request: Request) -> JSONResponse:
    """Direct API verification against a supplied 128-number enrolled faceprint."""
    configured_key = os.getenv("FACE_API_KEY")
    supplied_key = request.headers.get("X-API-Key")
    if not configured_key or not supplied_key or supplied_key != configured_key:
        return _error(401, "Unauthorized")
    form, error = await _read_form(request)
    if error:
        return error
    faceprint = _parse_faceprint(form.get("face_data"))
    if isinstance(faceprint, JSONResponse):
        return faceprint
    status_code, content, _ = await _process_capture(form, faceprint)
    return JSONResponse(status_code=status_code, content=content)
