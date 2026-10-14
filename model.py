from ultralytics import YOLO
import os

def train_model():
    # 1. Load the model weights
    # Using 'yolov12n.pt' is better than '.yaml' because it uses pre-trained knowledge
    model = YOLO(r'C:\Users\jeyaagastin\OneDrive\Desktop\new_KFS\yolov12n.pt')

    # 2. Start the training process
    model.train(
        # File paths
        data=r'C:\Users\jeyaagastin\OneDrive\Desktop\new_KFS\data.yaml', 
        name='weight',
        
        # Training length and hardware
        epochs=100,      # High accuracy training
        batch=16,        # Optimized for RTX 2050 (4GB VRAM)
        imgsz=640, 
        device=0,        # Uses your single NVIDIA GPU
        
        # High-performance augmentations from the previous code
        scale=0.5,       # Randomly zoom images
        mosaic=1.0,      # Combine 4 images into 1 (very effective)
        mixup=0.0,       # Keeps training stable for Nano models
        copy_paste=0.1,  # Useful for learning object outlines
    )

if __name__ == '__main__':
    # Professional practice: ensures the script runs correctly on Windows
    train_model()