# =========================
# IMPORT LIBRARIES
# =========================
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# DATASET PATH
# =========================
dataset_path = "dataset"

# =========================
# FEATURE FUNCTIONS
# =========================

def extract_color_features(image):
    return np.array(cv2.mean(image)[:3])

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist

# =========================
# LOAD DATA + PREPROCESS + FEATURES
# =========================
X = []
y = []
label_names = os.listdir(dataset_path)

for label, folder in enumerate(label_names):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # -------- PREPROCESSING --------
        img = cv2.resize(img, (128, 128))  # Resize
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Denoise

        # Enhance contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # -------- FEATURE EXTRACTION --------
        color_feat = extract_color_features(img)
        lbp_feat = extract_lbp(img)

        features = np.hstack([color_feat, lbp_feat])

        X.append(features)
        y.append(label)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

# Accuracy
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report (Precision, Recall, F1 Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# MANUAL TEST PREDICTION
# =========================

# 👉 User gives input image
test_image_path = input("\nEnter image path: ")

test_img = cv2.imread(test_image_path)

if test_img is None:
    print("❌ Image not found. Check path.")
else:
    # SAME preprocessing
    test_img = cv2.resize(test_img, (128, 128))
    test_img = cv2.GaussianBlur(test_img, (5, 5), 0)

    lab = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    test_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Extract features
    color_feat = extract_color_features(test_img)
    lbp_feat = extract_lbp(test_img)

    features = np.hstack([color_feat, lbp_feat]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    print("\nPredicted Bird:", label_names[prediction[0]])