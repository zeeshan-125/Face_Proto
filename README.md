# Face verification prototype

This repository supports separate **onboarding** and **offboarding** flows. Onboarding checks a live capture and stores the averaged SFace faceprint locally under a Profile ID. Offboarding looks up that Profile ID and verifies a fresh capture against the saved onboarding faceprint.

## Run

Install the packages in `requirements.txt`, prepare the local environment file and model files as described below, then start the API. Local secrets, faceprint data, and model binaries are intentionally excluded from Git.

```powershell
uvicorn app:app --reload
```

Create your private `.env` from the example and replace the placeholder with a local key:

```powershell
Copy-Item .env.example .env
```

The app reads `.env` automatically at startup. Download the MediaPipe Face Landmarker model from the [official MediaPipe model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) and save it as `models/face_landmarker.task`. Download compatible YuNet and SFace ONNX models from [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), saving them as `models/face_detection_yunet_2026may.onnx` and `models/face_recognition_sface_2021dec.onnx` respectively, or update the paths in `.env` to match your model filenames.

Open the browser UI at [http://127.0.0.1:8000](http://127.0.0.1:8000), select onboarding or offboarding, and use the same Profile ID in both. Upload a video or choose **Capture with camera**. Camera capture shows a two-second ready cue followed by each selected task for four seconds, then submits the recorded clip through the selected workflow. The browser asks for camera access only after the user clicks **Enable camera**.

Onboarding stores the averaged 128-number faceprint in the local SQLite database `faceprints.sqlite3`; the raw video is not retained. The database is ignored by Git. An existing Profile ID cannot be overwritten by onboarding, preventing an accidental baseline replacement. Offboarding with an unknown Profile ID returns `profile_not_found`.

The model assets used by this prototype are the MediaPipe Face Landmarker [official model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task), OpenCV Zoo [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), and OpenCV Zoo [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Their local paths are set in `.env`; model binaries are excluded from Git. YuNet uses OpenCV 5's dynamic-shape model with `OPENCV_FORCE_DNN_ENGINE=4`.

MediaPipe Tasks provides the face landmarks and gesture blendshapes ([official API reference](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions)). If Windows Application Control blocks MediaPipe's native runtime, the administrator must allow that runtime or the app must run on a machine where it is permitted; the prototype cannot bypass that operating-system policy.

OpenCV's SFace pipeline aligns crops from YuNet landmarks and emits features through `FaceRecognizerSF` ([OpenCV example](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/demo.cpp)).

The direct `POST /verify` endpoint expects multipart form data with:

- `actions`: a JSON array containing `blink` and/or `smile`, for example `["blink","smile"]`
- `face_data`: a JSON array of exactly 128 numbers
- `video`: the uploaded video file
- `X-API-Key`: the key configured in `FACE_API_KEY`

The API reads `FACE_API_KEY` from the process environment or the project `.env` file. Direct `POST /verify` requests still need `X-API-Key`; the same-origin browser UI posts to `/ui/enroll` or `/ui/offboard`, which uses the server-side key without exposing it in the page. UI routes reject cross-origin requests. The `.env` file is ignored by Git.

The API returns 401 for a missing or incorrect direct API key, 422 for invalid fields or an unreadable video, 413 when the video exceeds 25 MiB, and 503 when a model path is not configured. A valid upload is written to a temporary file, analyzed frame by frame, and deleted when the request finishes. Verification failures from the ordered checks are returned as `200` JSON with `status: "rejected"` and the first failing `reason`. Successful onboarding returns `status: "enrolled"`; successful offboarding returns `status: "verified"`.

Step B resizes frames wider than 640 px, reports face counts plus eye blink and mouth smile blendshape scores, and retains the sharpest frame in each second with exactly one face and an eye-closed score below 0.3. The response includes score arrays and the selected frame indices/sharpness values; the selected image arrays remain available in `VideoAnalysis.selected_frames` for the face matching step. The scores are MediaPipe blendshape estimates, so their thresholds may need calibration against real capture conditions.

Step C detects a blink when the eye-closed score rises above 0.5 and later falls below 0.3. It detects a smile when the smile score stays above 0.6 for at least three consecutive single-face frames, emitting only once per smile run. The response lists detected events in time order and reports `challenge_passed` when the requested `actions` appear in that order; extra events are allowed. Frames without exactly one face do not provide gesture scores, and break a smile run.

Step D checks, in order: unreadable video, no-face frames above 30%, multiple-face frames above 10%, incomplete requested challenge, then fewer than three good selected photos. A good photo has a face at least 100 px wide, Laplacian sharpness at least 40, and mean grayscale brightness at least 40/255. The face-width rule comes from the prototype description; the blur and brightness cutoffs are initial heuristics and may need tuning for camera conditions.

YuNet's detected face and five landmarks align each selected image before SFace generates a 128-number embedding. Every pair within a clip must have cosine similarity at least 0.363 or the flow returns `not_same_person`. Onboarding stores the average embedding. Offboarding compares its average against that saved template; a score below 0.363 returns `not_matched`. These thresholds are prototype defaults and should be validated with consented enrollment and verification samples before real use.
