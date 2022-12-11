# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.utils import load_img
# import matplotlib.pyplot as plt
from keras.models import load_model
# from tensorflow.keras.utils import img_to_array
# from tensorflow.keras.preprocessing import image
# import os
import cv2
import numpy as np
# from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
# from keras.preprocessing.image import load_img, img_to_array

# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        # captures frame and returns boolean value and captured image
        ret, test_img = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h),
                          (255, 0, 0), thickness=7)
            # cropping region of interest i.e. face area from  image
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            # img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.asarray(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            # img_pixels /= 255
            img_pixels = img_pixels / 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy',
                        'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()
        return frame

    # resized_img = cv2.resize(test_img, (1000, 700))
    # cv2.imshow('Facial Emotion Detection', resized_img)

#     if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
#         break

# cap.release()
# cv2.destroyAllWindows
