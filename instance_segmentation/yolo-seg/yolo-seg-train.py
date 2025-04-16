from ultralytics import YOLO

# Load a pre-trained YOLOv8-seg model
model = YOLO(r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\checkpoints\yolov8-seg\yolov8s-seg.pt')  # or 'yolov8n-seg.pt' for the nano version

# Train the model
model.train(
    data=r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\yolo_lp_screen_id\data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,  # Set to 'cpu' or '0' for GPU
    save=True,
    save_period=10,
    project="privacy-segmentation",
    name="yolov8s-finetune-balanced"
)

# Evaluate the model
metrics = model.val()  # Returns a dictionary with evaluation metrics

# Print the mean Average Precision (mAP) for the validation set
print(f"mAP: {metrics['map']}")

# Export the model to ONNX format
model.export(format='onnx')