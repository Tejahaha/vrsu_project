from ultralytics import YOLO

def main():
    print("Initializing YOLOv8m (Medium) model for highest speed-accuracy trade-off...")
    model = YOLO("yolov8m.pt")
    
    print("Starting training on RTX 5050 (GPU 0)...")
    results = model.train(
        data="c:/Users/TEJA/PycharmProjects/TumorFinder/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8, # Conservative for 8GB VRAM
        workers=4,
        device=0,
        project="BrainTumorDetection",
        name="yolov8m_training",
        exist_ok=True
    )
    
    print("Training Complete! The best weights are saved in 'BrainTumorDetection/yolov8m_training/weights/best.pt'")

if __name__ == "__main__":
    main()
