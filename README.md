# MiniVision - Fish Object Detection

MiniVision is a computer vision project that detects and classifies fish in images using a custom-trained YOLO11 object detection model.

## Features

- Detects fish in images.
- Predicts class labels and confidence scores.
- Draws bounding boxes around detected objects.
- Saves annotated images for inspection.
- Uses trained model weights stored in `models/best.pt`.

## Technologies Used

- Python
- Ultralytics YOLO11
- PyTorch
- OpenCV

## Project Structure

MiniVision-Object-Detection/
|-- dataset/
|   |-- train/
|   |-- valid/
|   |-- test/
|   `-- data.yaml
|-- models/
|   `-- best.pt
|-- main.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md

## Installation

Python and Git should be installed on your computer.

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/DikshaGul/MiniVison.git
cd MiniVison
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running MiniVision

Run the application from the project root:

```powershell
python main.py
```

Enter the path to an image when prompted. Press Enter to use the default test image included in the dataset.

The program displays the number of detected objects, their predicted classes, and their confidence scores. Annotated images are saved under the generated `runs` directory.

## Dataset

### Fish > 416x416

Source: https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml

Provided by Jacob Solawetz  
License: Public Domain

### Overview

![Image example](https://i.imgur.com/9hvxcRQ.jpg)

This is an object detection dataset of ocean fish classified by their Latin names.

![Image example](https://i.imgur.com/ECPln18.jpg)

### Use Cases

This dataset can be used for the following purposes:

- Underwater object detection model
- Fish object detection model
- Training an object detection model to recognize underwater species
- Prototyping a fish detection system
- Identifying fish with computer vision
- Fish bounding boxes and species annotations

These images have been listed in the public domain.

**Note:** The dataset description states that the images were sourced from makeml.app/datasets/fish.

MiniVision uses the training, validation, and test splits to train and evaluate its fish detection model.

## Model Training

The model was trained using Ultralytics YOLO11 with the following configuration:

| Setting | Value |
|---|---|
| Image size | 416 Ã— 416 |
| Batch size | 4 |
| Training epochs | 20 |
| Device | CPU |
| Number of configured classes | 26 |

The selected model weights are stored in `models/best.pt`.

## Evaluation Results

Validation of the selected best checkpoint reported the following approximate metrics:

| Metric | Result |
|---|---:|
| Precision | 39.3% |
| Recall | 53.2% |
| mAP@0.50 | 29.2% |
| mAP@0.50:0.95 | 21.4% |

These are initial results. The model demonstrates that the training and inference workflow operates, but detection performance still needs improvement. The metrics do not guarantee correct predictions on every image.

## Current Status

- Dataset downloaded and organized.
- Invalid zero-width or zero-height annotation rows identified and removed.
- YOLO11 model trained for 20 epochs.
- Trained weights saved locally.
- Image inference and annotated prediction saving verified.

## Future Improvements

- Review and consolidate overlapping or duplicate class labels.
- Evaluate predictions across the full test set.
- Improve detection performance through dataset review and further experiments.




- Add a more user-friendly interface for image selection.
- Track evaluation metrics across training runs.

## Acknowledgements

The fish dataset is available through Roboflow Universe:
https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml

The dataset page identifies the license as Public Domain. Refer to the original source for dataset details and attribution.

MiniVision uses the Ultralytics YOLO framework for object detection.


