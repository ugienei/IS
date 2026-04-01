from ultralytics import YOLO

# โหลดโมเดล
def load_yolo():
    try:
        return YOLO("models/best_yolo.pt")
    except:
        return YOLO("yolov8n.pt")

# detect function
def detect(model, img, conf):
    results = model(img, conf=conf, imgsz=640)

    # fallback
    if len(results[0].boxes) == 0:
        results = model(img, conf=0.2, imgsz=640)

    return results