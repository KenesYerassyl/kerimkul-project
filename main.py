import cv2
import dlib
import numpy as np
from imutils import face_utils


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def setup():
    global face_cascade, predictor, lStart, lEnd, rStart, rEnd, cap, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, COUNTER, ALARM_ON

    face_cascade = cv2.CascadeClassifier(
        "./materials/haarcascade_frontalface_default.xml"
    )
    predictor = dlib.shape_predictor(
        "./materials/shape_predictor_68_face_landmarks.dat"
    )

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    COUNTER = 0
    ALARM_ON = False


def draw():
    global COUNTER, ALARM_ON

    ret, frame = cap.read()
    if not ret:
        return True

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    # Add your alarm code here (e.g., sound an alarm, send a notification)
                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(
            frame,
            "EAR: {:.2f}".format(ear),
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True

    return False


if __name__ == "__main__":
    setup()
    while True:
        if draw():
            cap.release()
            cv2.destroyAllWindows()
            break
