# TumorFinder 🧠

TumorFinder is a deep learning computer vision project leveraging Ultralytics YOLOv8 for detecting and classifying brain tumors in MRI scans. The project uses a medium-sized YOLO architecture (`yolov8m`) to balance inference speed with detection accuracy.

## Project Structure

Here is a breakdown of the core scripts in the repository:

- `data.yaml`: Configuration file for YOLO indicating paths to training, validation, and testing image sets. It also defines the 3 tumor categories.
- `remap_labels.py`: A utility script used to re-index dataset class labels so they map correctly to a 0-indexed YOLO format.
- `train.py`: The executable script to fine-tune the YOLOv8m model on the Custom Brain Tumor dataset.
- `visualize_attention.py`: A visualization script that performs inference and extracts internal feature maps. It produces an attention heatmap showing *where* the model is "looking" to make its prediction.

## Usage Guide

Follow these steps to correctly initialize and use the project.

### 1. Setup Data Labels
If your dataset labels are 1-indexed (e.g., classes 1, 2, 3), run the remap formatting script so YOLO recognizes them properly (0, 1, 2):
```bash
python remap_labels.py
```

### 2. Train the Model
Make sure your dataset is situated at `dataset_brain tumor_correct/`, exactly as designated inside `data.yaml`.
Then, start the training phase:
```bash
python train.py
```
This will train a YOLOv8m model for 50 epochs on GPU 0.
Upon completion, the best weights are saved at: `BrainTumorDetection/yolov8m_training/weights/best.pt`

### 3. Inference and Heatmaps
Want to visualize your model's accuracy on unseen testing images? 
Verify the weights are generated from Step 2, then run the visualization tool:
```bash
python visualize_attention.py
```
This script will produce a dual-plot image named `heatmap_output.png` containing:
1. The standard YOLO bounding box predictions.
2. A Gradient Activation Heatmap highlighting the tumor features the deep-learning network focused heavily on.
