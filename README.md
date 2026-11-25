# YOLO26 Target Detection

A small computer-vision project that fine-tunes an Ultralytics YOLO26 model to find the red circular targets in the included **Dot Pop** browser game. It includes a Roboflow-exported dataset, training and screen-inference scripts, pretrained weights, and an optional Windows mouse clicker.

> **Use responsibly:** `target_clicker.py` can move the mouse and send real left-clicks. Start with its debug mode enabled and use it only in this project's local demo or another environment where automation is allowed.

## What is in this project

| Path | Purpose |
| --- | --- |
| `game/index.html` | Local Dot Pop target-clicking demo. |
| `collection/` | The 10 original target screenshots in this dataset version. |
| `dataset/` | Images, annotations, split configuration, and Roboflow export notes. |
| `train.py` | Fine-tunes the pretrained `yolo26s.pt` checkpoint. |
| `screen_detector.py` | Runs detection on a selected screen and displays boxes. |
| `target_clicker.py` | Detects targets and optionally moves/clicks the mouse. |
| `yolo26n.pt`, `yolo26s.pt` | Included pretrained YOLO26 nano and small checkpoints. |
| `runs/detect/` | Training outputs, metrics, visualizations, and saved weights. |

## How YOLO works

YOLO means **You Only Look Once**. It is an object-detection approach that processes an image in one model pass rather than repeatedly scanning separate image regions.

1. **Input:** A screenshot or image is resized to the model's input size. This project trains with `imgsz=640`.
2. **Feature extraction:** The neural network learns visual patterns such as edges, colors, shapes, and object context.
3. **Predictions:** For each possible object, the model predicts a class and location. A detection includes a class, confidence score, and bounding box.
4. **Filtering:** Low-confidence predictions are discarded and overlapping predictions are handled by the model's post-processing.
5. **Output:** The remaining detections can be drawn on the image or used by another program.

### YOLO26 in this project

This repository uses Ultralytics' **YOLO26** pretrained checkpoint, `yolo26s.pt`, and fine-tunes it for one custom class: `Target-Detection`. The included `yolo26n.pt` is a smaller alternative checkpoint; `train.py` currently selects the small (`s`) checkpoint.

Training starts from pretrained weights rather than learning all visual features from scratch. This is called **transfer learning**. The resulting model is a custom detector for the examples in this dataset; it is not a general-purpose detector for arbitrary objects.

## End-to-end workflow

The workflow is:

1. Collect varied screenshots of the targets.
2. Annotate each target in Roboflow using one consistent class.
3. Review annotations, create train/validation/test splits, and export in YOLO format.
4. Train from the pretrained YOLO26 checkpoint.
5. Review validation and test results, then improve the dataset where the model fails.
6. Run still-image or screen inference. Use the optional clicker only in a permitted, controlled demo.

## 1. Collect images

The current dataset is based on screenshots of red targets on the Dot Pop game's dark grid. The `collection/` folder holds the original 10 screenshots used for this version.

For a stronger model, collect additional screenshots that vary in:

- Target position, size, and number of targets in a frame.
- Game level, screen resolution, and browser/window layout.
- Target appearances and the backgrounds or visual effects the detector may encounter.
- Frames with **no target**, if empty-screen false detections are a problem.

Use actual, varied frames rather than many near-identical copies. Keep a separate untouched test set and avoid placing near-duplicate frames in different splits; otherwise validation/test scores can be misleading. Only collect images you have permission to use.

### Run the included game for sample images

From the project directory, start a local web server:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000
```

Open `http://localhost:8000/game/` in a browser, start Dot Pop, and capture screenshots containing different target positions and sizes. Stop the server with `Ctrl+C`.

## 2. Annotate in Roboflow

The existing dataset came from the Roboflow project [Target_Detection](https://universe.roboflow.com/mahivs-workspace/target_detection-evks8). The exported dataset metadata records **CC BY 4.0**; retain the attribution and license when sharing or reusing the dataset.

For a new dataset version:

1. Create or open a Roboflow project for target detection and upload the collected screenshots.
2. Use the single class name `Target-Detection` consistently.
3. Mark every visible target once. Keep the annotation tight around the target, include the whole target, and do not label unrelated UI elements.
4. Review the annotations for missed, duplicate, or inconsistent labels before generating a version.
5. Create train, validation, and test splits. Keep related/near-duplicate frames together in one split.
6. For the existing export recipe, use auto-orientation and resize to `512x512` with **Stretch**; no Roboflow augmentations were applied to version 1.
7. Export/download the dataset in the Ultralytics YOLO format and place the exported split folders under `dataset/`.

### Annotation format in this repository

`dataset/data.yaml` defines one class, `Target-Detection`. The label files contain class ID `0` followed by normalized polygon points (values between `0` and `1`), not just the five values of a basic YOLO bounding-box row. The Roboflow export notes describe this as YOLOv8 format. Ultralytics accepts these polygon annotations for this detection training workflow and derives the enclosing detection boxes.

Do not mix annotation types or change class ordering between dataset versions. If you change the classes, update `names` and the class IDs in every label consistently.

## 3. Dataset preprocessing and layout

The current Roboflow export applies:

- Auto-orientation and removal of EXIF orientation metadata.
- Resize to `512x512` using Stretch.
- No export-time image augmentation.

The training script separately uses `imgsz=640`; Ultralytics prepares images for this model input during training. The training run also uses Ultralytics' configured training augmentation defaults, recorded in each run's `args.yaml`.

Expected project layout:

```text
target_detection/
|-- collection/
|-- dataset/
|   |-- data.yaml
|   |-- train/
|   |   |-- images/
|   |   `-- labels/
|   |-- valid/
|   |   |-- images/
|   |   `-- labels/
|   `-- test/
|       |-- images/
|       `-- labels/
|-- game/
|-- train.py
|-- screen_detector.py
`-- target_clicker.py
```

The checked-in version currently has **10 images: 6 train, 3 validation, and 1 test**, all for one class. This is enough to demonstrate the pipeline, not enough to establish reliable real-world accuracy. In particular, a test set of one image cannot give a meaningful estimate of generalization.

`dataset/data.yaml` is the source of the class and split paths. Before training a replacement export, confirm that it still points to the `train/images`, `valid/images`, and `test/images` directories and that each image has its corresponding label file.

## 4. Install dependencies

This project targets Windows. The checked workspace ran Python `3.14.7` with Ultralytics `8.4.218`; use a Python version supported by the PyTorch/Ultralytics packages available for your machine.

Create a virtual environment and install the packages used by the scripts:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ultralytics opencv-python mss numpy
```

Packages:

| Package | Used for |
| --- | --- |
| `ultralytics` | YOLO26 model loading, training, validation, and prediction. It installs PyTorch and other required dependencies. |
| `opencv-python` | Image color conversion, drawing, and the detector preview window. Imported as `cv2`. |
| `mss` | Capturing a monitor screenshot. |
| `numpy` | Handling captured image arrays. |
| Python standard library (`ctypes`, `time`) | Windows mouse/keyboard input and timing in `target_clicker.py`; no separate install is needed. |

The training script is configured for CPU (`device="cpu"`). This works without CUDA but can be slow. To train on an NVIDIA GPU, install a PyTorch build compatible with the machine's CUDA setup, then change `device` in `train.py` to the appropriate GPU index (usually `0`).

## 5. Train the model

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe train.py
```

The current `train.py` configuration is:

| Setting | Value |
| --- | --- |
| Initial weights | `yolo26s.pt` |
| Dataset | `dataset/data.yaml` |
| Epochs | `100` |
| Image size | `640` |
| Batch size | `4` |
| Device | CPU |
| Workers | `0` |

Ultralytics prints the run's save directory. Each run saves its exact settings in `args.yaml`; consult that file rather than assuming an automatically incremented run name. The existing trained run is under `runs/detect/runs/detect/target_detection-8/` and contains:

Only the current `weights/best.pt` checkpoint is kept tracked in Git. Other run outputs (plots, logs, older checkpoints, and new training runs) are ignored; they remain local and can be regenerated by training.

- `weights/best.pt` — checkpoint with the best validation result during training.
- `weights/last.pt` — checkpoint from the final epoch.
- `results.csv` and `results.png` — training/validation metric history.
- `confusion_matrix.png`, `BoxPR_curve.png`, `BoxP_curve.png`, and `BoxR_curve.png` — evaluation plots.
- `train_batch*.jpg` and `val_batch*_*.jpg` — examples of training labels and validation predictions.

The detector scripts currently point to:

```text
runs\detect\runs\detect\target_detection-8\weights\best.pt
```

After training again, update `MODEL_PATH` in **both** `screen_detector.py` and `target_clicker.py` to the `best.pt` path printed/saved for the new run. Run names may be automatically incremented.

## 6. Evaluate and inspect predictions

Review the validation curves and compare `val_batch0_labels.jpg` with `val_batch0_pred.jpg`. Look for missed targets, incorrect boxes, and false detections. Do not select or tune the model using the test images; use the test split once for a final check.

To evaluate the held-out test split with the current checkpoint:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO(r'runs\detect\runs\detect\target_detection-8\weights\best.pt'); r=m.val(data=r'dataset\data.yaml', split='test'); print(r.results_dict)"
```

Useful measures include precision (how many predicted targets are correct), recall (how many labeled targets were found), and mAP (detection quality across confidence/overlap thresholds). The current one-image test split is too small for dependable conclusions. Add diverse labeled data and keep the test set separate before relying on these metrics.

For a still-image prediction, point YOLO at an image or directory:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'runs\detect\runs\detect\target_detection-8\weights\best.pt').predict(source=r'dataset\test\images', conf=0.25, imgsz=640, save=True)"
```

Ultralytics saves the rendered prediction in the run directory it reports.

## 7. Run live screen detection

After confirming the model path, run:

```powershell
.\.venv\Scripts\python.exe screen_detector.py
```

`screen_detector.py` captures monitor index `1`, runs prediction at confidence `0.25` and image size `640`, and displays the detected boxes. Press **Q** in the preview window to stop. If the wrong display is captured, change `monitor = sct.monitors[1]` to the desired monitor index. In `mss`, index `0` represents the combined virtual desktop; the other indexes refer to individual monitors.

## 8. Optional mouse clicker

`target_clicker.py` uses the same screen capture and model inference, chooses a detection, moves the Windows cursor to its center, and can send a left-click. It does not start the browser game for you.

1. Set `DEBUG_MODE = True` in `target_clicker.py` first. In this mode it only moves the cursor; it does not click.
2. Check that the selected monitor and `MODEL_PATH` are correct.
3. Run it against the local Dot Pop demo:

   ```powershell
   .\.venv\Scripts\python.exe target_clicker.py
   ```

4. Press **Esc** to stop.
5. Only if you intentionally want real clicks in an environment where automation is permitted, set `DEBUG_MODE = False`.

Other options near the top of the file:

- `CONF`: minimum detection confidence (`0.22` by default).
- `PICK_MODE`: `"confidence"` chooses the most confident target; `"center"` chooses the target nearest the screen center.
- `CLICK_COOLDOWN`: minimum delay between clicks.
- `CLICK_HOLD`: how long the left button is held.
- `MONITOR_INDEX`: display captured by the script.

## Use cases and limits

This project is suitable as a learning/demo pipeline for:

- Detecting a known visual target in screenshots.
- Practicing image collection, annotation, transfer learning, and evaluation.
- Demonstrating screen-based inference in the included local browser game.
- Prototyping user-interface testing in applications where automation is explicitly allowed.

It is **not** evidence that the model will detect other object types or work reliably in different games, displays, or real-world scenes. Those tasks need their own representative dataset, annotation policy, evaluation, and permission to automate the target application.

## Common issues

| Issue | What to check |
| --- | --- |
| `best.pt` not found | Use the exact run directory printed by Ultralytics and update `MODEL_PATH` in both screen scripts. |
| Training cannot find images | Check `dataset/data.yaml` and confirm all three split image folders exist. |
| Labels are missing or mismatched | Each image must have a matching `.txt` label file; verify Roboflow class IDs and class ordering. |
| No detections | Confirm the loaded checkpoint, monitor index, target visibility, and confidence threshold. Test a saved screenshot before live capture. |
| Training is slow | CPU training is configured by default. Use a compatible CUDA-enabled PyTorch setup and set `device` appropriately if an NVIDIA GPU is available. |
| Clicker acts on the wrong screen | Set `MONITOR_INDEX` correctly and leave `DEBUG_MODE = True` until the cursor target has been verified. |
