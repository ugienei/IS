"""
train_yolo.py
- เทรนโมเดล YOLOv8 ด้วยชุดข้อมูล GTSRB
"""
from ultralytics import YOLO

PRETRAINED_MODEL = 'runs/detect/gtsrb_yolo_v8_19/weights/best.pt'

DATA_CONFIG_PATH = 'data/gtsrb-yolo.yaml'
 
# การตั้งค่าการเทรน
EPOCHS = 15        
IMG_SIZE = 416    
BATCH_SIZE = 4

def main():
    print("Loading pre-trained model...")
    model = YOLO(PRETRAINED_MODEL)

    print(f"Starting training on '{DATA_CONFIG_PATH}' for {EPOCHS} epochs...")
    model.train(
        data=DATA_CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='gtsrb_yolo_v8_20'  
    )

    print("\n✅ Training complete!")
    print("Your trained model is saved in the 'runs/detect/gtsrb_yolo_v8/' folder.")
    print("Look for the file 'best.pt' inside 'runs/detect/gtsrb_yolo_v8/weights/'")

if __name__ == '__main__':
    main()