import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("faceexpression_data.npy")

print(data.shape, data.dtype)

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        nose = landmarks.parts()[27]
        # print(nose.x, nose.y)
        # for point in landmarks.parts()[17:]:
        #     expression = np.array([[point.x - face.left(), point.y - face.top()]])
        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])
        mood = model.predict(expression.flatten().reshape(1, -1))
        print(mood[0])
            # cv2.putText(frame, mood[0], (face.left()+10,face.top()+10 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)


    # print(faces)

    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()