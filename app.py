
import cv2
import os
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

app = Flask(__name__)
load_dotenv()

# 1. Load Resources
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# model = load_model('mask_detector.model') # Ensure model is in the same folder

def send_email_alert():
    msg = EmailMessage()
    msg.set_content("Face Mask Violation Detected!")
    msg['Subject'] = 'SECURITY ALERT'
    msg['From'] = os.getenv('EMAIL_USER')
    msg['To'] = os.getenv('EMAIL_RECEIVER')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASS'))
            server.send_message(msg)
    except: pass

# 1. Load Resources
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mask_model_path = os.path.join(os.path.dirname(__file__), 'mask_detector.model')
model = load_model(mask_model_path) if os.path.exists(mask_model_path) else None


def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), (224, 224))
            face_img = img_to_array(face_img) / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            label = "Mask"
            color = (0, 255, 0)

            if model is not None:
                pred = model.predict(face_img, verbose=0)
                (mask_prob, without_mask_prob) = pred[0]
                if without_mask_prob > mask_prob:
                    label = "No Mask"
                    color = (0, 0, 255)
                    cv2.putText(frame, "ALERT", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # send_email_alert()

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
