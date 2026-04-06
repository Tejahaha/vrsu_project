import cv2
import torch
import numpy as np
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt

def get_attention_map(image_path, model_path):
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image.")
        return
        
    # Resize to YOLO default 640x640 for consistent feature map scaling
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    tensor_img = torch.from_numpy(img_rgb).float() / 255.0
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(model.device)
    
    activations = []
    # Hook into the second to last layer (typically the final C2f before the detection head)
    # This provides a rich spatial feature map representing what the network is "looking" at.
    target_layer = model.model.model[-2]
    
    def hook_fn(m, i, o):
        activations.append(o)
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Run YOLO inference
    print("Extracting feature map and running inference...")
    with torch.no_grad():
        preds = model(image_path) # native inference to get the plotted box
        model.model(tensor_img)   # tensor inference to trigger the hook
        
    handle.remove()
    
    if len(activations) == 0:
        print("Error: No activation captured.")
        return
        
    # Compute activation map by averaging across channels (simulates EigenCAM)
    act = activations[0][0].cpu().numpy()
    heatmap = np.mean(act, axis=0)
    
    # Normalize Heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.5, heatmap_colored, 0.5, 0)
    
    # Setup dual plot (Detection vs Attention)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("YOLO Bounding Box Prediction")
    res_plotted = preds[0].plot()
    res_plotted = cv2.cvtColor(cv2.resize(res_plotted, (640, 640)), cv2.BGR2RGB)
    plt.imshow(res_plotted)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Model Attention Heatmap (Feature Map)")
    plt.imshow(cv2.cvtColor(overlay, cv2.BGR2RGB))
    plt.axis("off")
    
    plt.tight_layout()
    output_path = "heatmap_output.png"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Saved visualization to '{output_path}'")

if __name__ == "__main__":
    # Automatically pick the first image in the Test set
    test_images = glob.glob(r"c:\Users\TEJA\PycharmProjects\TumorFinder\dataset_brain tumor_correct\Test\Images\*.jpg")
    best_weights = "BrainTumorDetection/yolov8m_training/weights/best.pt"
    
    import os
    if not os.path.exists(best_weights):
         print(f"Could not find trained weights at {best_weights}. Please run train.py first!")
    elif test_images:
        get_attention_map(test_images[0], best_weights)
    else:
        print("No test images found in the Dataset folder.")
