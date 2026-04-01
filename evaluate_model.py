from ultralytics import YOLO

MODEL = r'c:/CS461/runs/detect/gtsrb_yolo_v8_14/weights/best.pt'
DATA = r'c:/CS461/data/gtsrb-yolo.yaml'
IMGSZ = 320

print(f"Evaluating model: {MODEL}\nData: {DATA}\nimgsz: {IMGSZ}\n")

try:
    model = YOLO(MODEL)
    results = model.val(data=DATA, imgsz=IMGSZ)
    print('\nValidation finished.')
    print('Results object:', results)
except Exception as e:
    print('Evaluation failed:', e)
