import tensorflow as tf
import os
import numpy as np
import cv2
from PIL import Image


def recognize(gray):
    emotion_strings = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    images = []
    cv2.imwrite("gray.png", gray)
    pil_image = Image.open("gray.png").convert("L").resize((48, 48), Image.ANTIALIAS)
    pixels = np.array(pil_image, "uint8")
    images.append(pixels)
    images = np.array(images)
    images = images.astype("float32") / 255.0

    x_input = np.empty([int(len(images)), 48, 48, 3])
    for i, item in enumerate(x_input):
        item[:, :, 0] = images[i]
        item[:, :, 1] = images[i]
        item[:, :, 2] = images[i]

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "../model")
        predictions = sess.run('sequential_1/dense_4/Softmax:0', feed_dict={'input_2:0': x_input})

        print("Probability distribution:")
        for i in range(0, len(predictions[0])):
            print(emotion_strings[i], ":", predictions[0][i] * 100, "%")
        print()


def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            if cv2.waitKey(1) & 0xFF == ord('k'):
                recognize(roi_gray)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Emotion detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    os.remove("gray.png")


if __name__ == "__main__":
    main()
