import os
from dotenv import load_dotenv
from pathlib import Path
from ultralytics import YOLO
load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.pt"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "416"))
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.35")
)
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))
from pathlib import Path
from ultralytics import YOLO

# Load the trained MiniVision fish-detection model
MODEL_PATH = Path("models/best.pt")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

model = YOLO(str(MODEL_PATH))

# Ask the user which image to analyse
image_path = input(
    "Enter the image path (or press Enter for a test image): "
).strip().strip('"')

if not image_path:
    image_path = (
        "dataset/test/images/"
        "FishDataset119_png.rf.5400284ea33a58a61de4f0ae9c380c13.jpg"
    )

if not Path(image_path).is_file():
    raise FileNotFoundError(
        f"Image not found: {Path(image_path).resolve()}"
    )

# Detect fish and suppress overlapping boxes
results = model.predict(
    source=image_path,
   imgsz=IMAGE_SIZE,
conf=CONFIDENCE_THRESHOLD,
iou=IOU_THRESHOLD,
    agnostic_nms=True,
    save=True,
    project="runs/minivision",
    name="prediction",
    exist_ok=True
)

# Display detection summary
result = results[0]
print(f"\nImage: {image_path}")
print(f"Objects detected: {len(result.boxes)}")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
    print(f"- {class_name}: {confidence:.2%} confidence")

print(f"Annotated image saved in: {result.save_dir}")
