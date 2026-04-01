"""
prepare_data.py
- อ่านไฟล์ GTSRB (สมมติเป็น folder หรือ csv)
- ปรับขนาดภาพ และจัดโครงสร้างเป็น ImageFolder: data/train/<class>/*.ppm
- แบ่ง train/val/test (stratified split)
"""
import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# CONFIG
RAW_DIR = "data/GTSRB"     # ปรับตามตำแหน่งที่เก็บ
OUT_DIR = "data"           # จะสร้าง data/train, data/val, data/test
IMG_SIZE = (48, 48)
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_STATE = 42

def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def save_image(img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

def main():
    # ตัวอย่าง: ถ้า RAW_DIR เป็น structure แบบ GTSRB ที่มี csv labels:
    # สมมติมีไฟล์ CSV: GT-final_test.csv หรือ Training/Labels...
    # เพื่อความยืดหยุ่น: ค้นหาทุกไฟล์ภาพใน RAW_DIR
    img_paths = []
    labels = []
    for ext in ("png","ppm","jpg","jpeg"):
        img_paths.extend(glob(os.path.join(RAW_DIR, "**", f"*.{ext}"), recursive=True))
    img_paths = sorted(img_paths)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {RAW_DIR} - กรุณาแตกไฟล์ GTSRB ลงที่ folder นี้")

    # Heuristic: class id เป็นชื่อ folder พ่อแม่ (ถ้ามาจัดโฟลเดอร์ per-class)
    for p in img_paths:
        # parent folder name
        cls = os.path.basename(os.path.dirname(p))
        labels.append(cls)

    df = pd.DataFrame({"path": img_paths, "label": labels})
    # encode labels to string (บางกรณี label เป็นตัวเลขในชื่อ)
    df['label'] = df['label'].astype(str)

    # split: first split out test, then split train->val
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(train_val_df, test_size=VAL_SIZE/(1-TEST_SIZE), stratify=train_val_df['label'], random_state=RANDOM_STATE)

    # prepare output folders
    for d in ["train", "val", "test"]:
        ensure_clean_dir(os.path.join(OUT_DIR, d))

    # helper to copy & resize
    def process_and_copy(df_in, split_name):
        for idx, row in tqdm(df_in.reset_index(drop=True).iterrows(), total=len(df_in), desc=f"Processing {split_name}"):
            src = row['path']
            lbl = str(row['label'])
            # read image (BGR)
            img = cv2.imread(src)
            if img is None:
                continue
            # resize to IMG_SIZE
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            # save to data/<split>/<label>/
            out_path = os.path.join(OUT_DIR, split_name, lbl, f"{idx}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, img)

    process_and_copy(train_df, "train")
    process_and_copy(val_df, "val")
    process_and_copy(test_df, "test")
    print("Data preparation done. Folders: data/train, data/val, data/test")

if __name__ == "__main__":
    main()
