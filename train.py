from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO('runs/detect/train18/weights/best.pt')

# Use the model
results = model.train(data="config.yaml", epochs=13, save = True, save_period = 3)  # train the model
