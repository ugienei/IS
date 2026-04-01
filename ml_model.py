import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

IMG_SIZE = 64
DATASET_PATH = "dataset"

def train_ml():
    X, y = [], []

    for label in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    svm = SVC(probability=True)
    knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    return rf, svm, knn

def predict(img, rf, svm, knn):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1)

    p1 = rf.predict(img)[0]
    p2 = svm.predict(img)[0]
    p3 = knn.predict(img)[0]

    votes = [p1, p2, p3]
    vote_count = Counter(votes)

    final = vote_count.most_common(1)[0][0]
    return final, vote_count