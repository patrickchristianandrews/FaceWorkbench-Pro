# FaceWorkbench Pro — expanded development build (v0.2)

**Not the full 97-feature master-specification application.** This build implements genuine, local computation for many of the core workflows, but it is not appropriate to claim every listed capability is implemented, independently validated, or usable without additional weights. In particular, **no Windows EXE is included**; Windows compilation is performed by `build_windows.bat` on a Windows PC.

## November 17 development update

- Windows GitHub Actions now packages a **single-file** `FaceWorkbench-Pro.exe` and makes it available from the successful workflow run's **Artifacts** section.
- Recognition supports configurable horizontal-flip embedding augmentation, per-model fusion weights, and independent session caching for multiple ONNX models.
- The **System & Model Registry** workspace exposes augmentation and per-model weighting controls.
- Studio offers secure-context webcam capture (localhost/HTTPS), analysis and image quality overlay.
- Gallery enrollment checks the probe against existing embeddings and prompts when it may be a duplicate. Similarity warnings **do not confirm identity**.
- Source and UI changes are not proof that model-specific inference and feature accuracy are validated. Only the installed compatible ONNX weights can provide recognition.

## Implemented

- 8-workspace responsive dark UI and FastAPI + PyWebView native launcher.
- Local OpenCV detection (fallback Haar and optional YuNet), face crop storage, quality gates, SQLite people and samples, rename / merge / delete.
- Face embedding inference using OpenCV SFace or compatible ArcFace-like ONNX output pipelines, embedding storage, L2 normalization, optional mirror augmentation, 1:1 similarity and 1:N gallery search / lookalike ranking, per-model similarity scores, and configurable weighted fusion. Detection-dependent five-point geometric alignment for supported detector output.
- SQLite embedding persistence, per-model FAISS exact inner-product index rebuild when `faiss-cpu` is installed, JSON/NPZ vector export, JSON vector import and backfill embeddings from saved crop images.
- Empirical same-identity versus different-identity ROC / TAR / FAR calculations, cosine threshold estimates, hard-negative pairs, and logistic stacker **training** with scikit-learn. This does not mean calibrated probabilities are available in verification responses.
- Greedy IoU bounding-box tracking, blur-all / mosaic and gallery-reference-based blur-matched / keep-matched video redaction. Audio is **not** carried into the MP4 output; visual detections must be reviewed for missed faces.
- Local EXIF/XMP inspector, external search launch links, folder enrollment through the native desktop picker, ZIP gallery backup, local PWA shell manifest and service worker.
- ONNX model registry, SFace and YuNet public fetch links, live threshold / model enable/disable / CPU-thread settings, missing-weight graceful skip.

## Not yet implemented or not validated

- SCRFD detection, dense 106/3D-68 landmark output and mesh, comprehensive 16-family model-specific adapters/verified model downloads. Many third-party weight files have no stable verified direct URL and may have separate terms and noncommercial licenses.
- Learned calibrated probability **inference**, operational biometric false positive verification, test-set FNIR/FPIR open-set statistics, reliable model-by-model accuracy comparison. The simple logistic stacker is trained on existing pairs but does not establish a calibrated match probability.
- Automatic discovery/storage of unknown-face crops and one-click promotion of unknown clusters. Clustering currently runs on saved face embeddings.
- Liveness, age/sex/emotion/mask classification, OpenFace AU telemetry, eye-gaze, 3D pose, OSNet or learned person ReID across cuts.
- Model-aware quality yaw gates and advanced behavior display, full identity-duplicate merge modal, real-time WebSocket process telemetry, authenticated LAN pairing and HTTPS certificate provisioning, all service-specific OSINT helpers.
- End-to-end usability, bundled model availability, and feature accuracy in the one-file Windows EXE are not fully validated. GitHub Actions compiles and runs automated smoke tests on Windows.

## Install and run (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python desktop.py
```

If a dependency fails to install, prefer a supported, pre-built wheel for Python 3.11. To use a working recognition path, open **System & Models**, download **YuNet** and **SFace**, enable SFace, select YuNet as detector and click Save. Subsequent enrollment records vectors; previously enrolled photos can be re-embedded using **Re-embed gallery crops** (lower quality than the original full image alignment). For other ONNX models you must supply legally acquired compatible weights under `models/`, then enable those models in Settings. Arbitrary model files can have incompatible input normalization, color order, output shape or license; testing is required.

To build the clickable Windows executable:

```powershell
build_windows.bat
```

The script creates `dist\FaceWorkbench\FaceWorkbench.exe`. Distribute the whole `FaceWorkbench` directory inside `dist`, not just the `.exe`. It builds on Windows because PyInstaller does not cross-compile from Linux.

Browser-only testing: `uvicorn app:app --host 127.0.0.1 --port 8000` then open <http://127.0.0.1:8000>. Run regression tests using `python -m pytest -q test_smoke.py`.

## Data and privacy

All gallery data is stored in `data/gallery.db` and `data/crops`; settings and downloaded model files are adjacent to the executable when frozen. Face crops and vector embeddings are sensitive biometric information. Only process images with an appropriate legal basis and protect or encrypt the device and any backup exports. Local API listens on loopback, not LAN. Image metadata may include GPS information. External OSINT links do not automatically upload images. Track-based redaction can miss undetected faces; inspect the entire result before disclosure.

**Important:** Cosine similarity is not a probability that two people are the same person. A threshold and an in-sample FAR cannot establish reliability on unseen subjects.
