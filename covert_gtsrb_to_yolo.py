"""
convert_gtsrb_to_yolo.py
- อ่านข้อมูล GTSRB (PPM + CSV)
- แปลงภาพเป็น JPG
- แปลงพิกัด (Bounding Box) ให้อยู่ใน format ของ YOLO (.txt)
- สร้างชุดข้อมูลสำหรับ YOLO ที่ path 'data/gtsrb_yolo/'
"""
import os
import pandas as pd
import cv2
from tqdm import tqdm

# --- 1. กำหนด Path ---
GTSRB_ROOT = "data/GTSRB"
OUTPUT_DIR = "data/gtsrb_yolo"

# Path ไปยังไฟล์ CSV ที่มีพิกัด
TRAIN_CSV_PATH = os.path.join(GTSRB_ROOT, "Train.csv")
TEST_CSV_PATH = os.path.join(GTSRB_ROOT, "Test.csv") # นี่คือไฟล์ label ของ Test set

# สร้างโฟลเดอร์ปลายทาง
os.makedirs(os.path.join(OUTPUT_DIR, "images/train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images/val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels/val"), exist_ok=True)

# --- 2. ฟังก์ชันแปลงพิกัด ---
def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """แปลง (x1,y1,x2,y2) เป็น (x_center, y_center, width, height) แบบ normalized"""
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    x_center = ((x1 + x2) / 2.0) * dw
    y_center = ((y1 + y2) / 2.0) * dh
    width = (x2 - x1) * dw
    height = (y2 - y1) * dh
    
    return (x_center, y_center, width, height)

# --- 3. ประมวลผลชุดข้อมูล Training ---
print("Processing Training Set...")
train_df = pd.read_csv(TRAIN_CSV_PATH)

for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    img_path = os.path.join(GTSRB_ROOT, row['Path'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
        
    img_height, img_width, _ = img.shape
    
    # ดึงข้อมูล
    class_id = row['ClassId']
    x1, y1 = row['Roi.X1'], row['Roi.Y1']
    x2, y2 = row['Roi.X2'], row['Roi.Y2']
    
    # แปลงพิกัด
    yolo_coords = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
    yolo_label_str = f"{class_id} {yolo_coords[0]} {yolo_coords[1]} {yolo_coords[2]} {yolo_coords[3]}\n"
    
    # สร้างชื่อไฟล์ (เช่น 00000_00000.ppm -> 0_00000_00000.jpg)
    base_filename = os.path.basename(row['Path']).replace(".ppm", "")
    class_folder = row['Path'].split('/')[1] # ได้ '0', '1', ...
    output_filename = f"{class_folder}_{base_filename}"
    
    # 1. บันทึกภาพเป็น JPG
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images/train", f"{output_filename}.jpg"), img)
    
    # 2. บันทึก Label .txt
    with open(os.path.join(OUTPUT_DIR, "labels/train", f"{output_filename}.txt"), "w") as f:
        f.write(yolo_label_str)

print("Training set processing complete.")

# --- 4. ประมวลผลชุดข้อมูล Validation (Test Set) ---
print("\nProcessing Validation (Test) Set...")
# เราต้องใช้ 'GT-final_test.csv' เพราะ 'Test.csv' ไม่มี label
test_df = pd.read_csv(TEST_CSV_PATH)

for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    img_path_relative = row['Path'] 
    img_filename = os.path.basename(img_path_relative)
    img_path = os.path.join(GTSRB_ROOT, img_path_relative)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
        
    img_height, img_width, _ = img.shape

    # ดึงข้อมูล
    class_id = row['ClassId']
    x1, y1 = row['Roi.X1'], row['Roi.Y1']
    x2, y2 = row['Roi.X2'], row['Roi.Y2']
    
    # แปลงพิกัด
    yolo_coords = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
    yolo_label_str = f"{class_id} {yolo_coords[0]} {yolo_coords[1]} {yolo_coords[2]} {yolo_coords[3]}\n"
    
    # สร้างชื่อไฟล์
    base_filename = img_filename.replace(".ppm", "")
    
    # 1. บันทึกภาพเป็น JPG
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images/val", f"{base_filename}.jpg"), img)
    
    # 2. บันทึก Label .txt
    with open(os.path.join(OUTPUT_DIR, "labels/val", f"{base_filename}.txt"), "w") as f:
        f.write(yolo_label_str)

print("Validation set processing complete.")
print(f"\n✅ Successfully converted dataset to YOLO format at: {OUTPUT_DIR}")