# AIML_P1
AI Face Mask Detection and Compliance Monitoring System is a real-time computer vision application that uses OpenCV and a CNN-based deep learning model to detect faces and classify whether a person is wearing a mask. The system displays “MASK ON”

AI Face Mask Detection & Compliance Monitoring System
📌 Project Overview

The AI Face Mask Detection and Compliance Monitoring System is a real-time computer vision application that uses Python, OpenCV, TensorFlow/Keras, CNN, and Flask to detect whether a person is wearing a face mask.

The system processes live webcam video, detects faces, classifies mask usage, assigns a temporary tracking ID, and counts only people who are wearing masks.

Core behavior
😷 Mask detected → MASK ON → Person is included in the compliance count.
⚠️ No mask detected → YOU HAVE TO WEAR A MASK! → Person is not included in the compliance count.
👤 Each tracked person receives a temporary Person ID.
💾 Face images can be automatically stored in separate dataset folders.
📊 Detection information is recorded in a CSV file.
🎯 Objectives
Detect faces in real time using a webcam.
Classify faces into:
with_mask
without_mask
Provide an immediate visual warning for people without masks.
Track different people using temporary IDs.
Count only currently detected masked people.
Store detected face images for dataset development.
Maintain structured detection records.
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
OpenCV	Webcam processing and face detection
TensorFlow / Keras	CNN model training and prediction
NumPy	Numerical and image processing
Scikit-learn	Dataset splitting and label processing
Flask	Web application and live video streaming
HTML/CSS	Web interface
CSV	Detection record storage
🧠 System Architecture
                    ┌─────────────────┐
                    │     Webcam      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Face Detection │
                    │     OpenCV      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Face Crop &   │
                    │  Preprocessing  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   CNN Model     │
                    │ TensorFlow/Keras│
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              ┌───────────┐     ┌────────────┐
              │  Mask     │     │  No Mask   │
              └─────┬─────┘     └──────┬─────┘
                    │                  │
                    ▼                  ▼
              "MASK ON"        "YOU HAVE TO
                    │            WEAR A MASK!"
                    │                  │
                    ▼                  ▼
              Count Masked       Not Counted
                 Person
                    │
                    ▼
             Dataset + CSV
📂 Project Structure
AIML_LIVE1/
│
├── train_model.py
│
├── flask_app/
│   │
│   ├── app.py
│   ├── mask_detector.keras
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── dataset/
│       │
│       ├── with_mask/
│       │
│       ├── without_mask/
│       │
│       └── person_records.csv
│
└── README.md
📁 Dataset

The training dataset contains two classes:

dataset/
│
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
with_mask

Contains images of people wearing masks.

without_mask

Contains images of people without masks.

🧠 CNN Model

The project uses a Convolutional Neural Network for binary image classification.

The model performs:

Input Image
     ↓
224 × 224 × 3
     ↓
Convolution
     ↓
Max Pooling
     ↓
Convolution
     ↓
Max Pooling
     ↓
Convolution
     ↓
Max Pooling
     ↓
Flatten
     ↓
Dropout
     ↓
Dense Layer
     ↓
Softmax
     ↓
Mask / No Mask

The final model produces two probabilities:

Mask Probability
No Mask Probability

The class with the higher probability determines the displayed status.

👤 Person ID Tracking

The system uses a centroid-based tracking mechanism to associate a temporary ID with a detected face.

Example:

Person ID: 1 → MASK ON
Person ID: 2 → YOU HAVE TO WEAR A MASK!
Person ID: 3 → MASK ON

The important point is that the ID is a temporary tracking ID, not permanent face recognition.

If a person leaves the camera view and the tracker loses them, they may receive a new ID when they return.

📊 Mask Compliance Counting

The project specifically counts masked people, rather than simply counting all detected faces.

Example:

Person 1 → MASK ON                    → ✅ Counted
Person 2 → YOU HAVE TO WEAR A MASK!  → ❌ Not counted
Person 3 → MASK ON                    → ✅ Counted

Therefore:

Total detected people = 3
Masked people = 2

If Person 1 removes the mask:

Person 1 → YOU HAVE TO WEAR A MASK!
Person 2 → YOU HAVE TO WEAR A MASK!
Person 3 → MASK ON

Then:

Masked people = 1

This makes the count represent mask-compliant people.

💾 Automatic Dataset Collection

The application can save detected face crops into:

dataset/
├── with_mask/
└── without_mask/

For example:

with_mask/
└── person_1_mask_20260917_213012.jpg

without_mask/
└── person_2_no_mask_20260917_213014.jpg

A time interval is used between saves so that the system does not create hundreds of identical images from the same webcam stream.

Important: Automatically collected images should be manually reviewed before being used for retraining because an incorrect prediction can create incorrectly labelled training data.

📋 CSV Records

The system can maintain:

person_records.csv

with information such as:

Field	Description
person_id	Temporary tracking ID
status	Mask / No Mask
mask_probability	Model probability for mask
no_mask_probability	Model probability for no mask
timestamp	Detection time
image_path	Saved image location

Example:

1,Mask,0.96,0.04,2026-09-17 21:30:12,with_mask/person_1_mask.jpg
2,No Mask,0.05,0.95,2026-09-17 21:30:15,without_mask/person_2_no_mask.jpg
🌐 Flask Web Application

The Flask application provides a browser-based live monitoring interface.

Run the application and open:

http://127.0.0.1:5000

The interface displays:

Live webcam stream
Face bounding boxes
Person ID
Mask status
Mask probability
No-mask probability
Mask compliance information
