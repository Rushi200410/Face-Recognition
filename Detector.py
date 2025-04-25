import cv2
from tkinter import messagebox
import os

def main_app():
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Load all trained models
    models = {}
    classifier_dir = "./data/classifiers/"
    for file in os.listdir(classifier_dir):
        if file.endswith("_classifier.xml"):
            name = file.replace("_classifier.xml", "")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(os.path.join(classifier_dir, file))
            models[name] = recognizer

    if not models:
        messagebox.showerror('Error', 'No trained models found.')
        return

    pred = False
    recognized_name = "Unknown"

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            best_confidence = 0
            best_name = "Unknown"

            for name, recognizer in models.items():
                try:
                    id_, confidence = recognizer.predict(roi_gray)
                    confidence = 100 - int(confidence)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_name = name
                except:
                    continue

            if best_confidence > 50:
                pred = True
                recognized_name = best_name
                text = f"Recognized: {recognized_name.upper()} ({best_confidence}%)"
                color = (0, 255, 0)
            else:
                text = "Unknown Face"
                color = (0, 0, 255)

            font = cv2.FONT_HERSHEY_PLAIN
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            frame = cv2.putText(frame, text, (x, y - 4), font, 1, color, 1, cv2.LINE_AA)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            print(pred)
            if pred:
                messagebox.showinfo('Success', f'{recognized_name} has been recognized')
            else:
                messagebox.showerror('Alert', 'No known user detected. Try again.')
            break

    cap.release()
    cv2.destroyAllWindows()
