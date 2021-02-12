# # Face Video Detection

import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# closed_frontal_palm = cv.CascadeClassifier('closed_frontal_palm.xml')
# palm = cv.CascadeClassifier('palm.xml')
# img = cv.imread('me.jpg')

cap = cv.VideoCapture(0)

while cap.isOpened():

    _, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # palm = palm.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)

    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()



# # Face Image Detection

# import cv2 as cv

# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv.imread('me.jpg')


# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# for (x, y, w, h) in faces:
#     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

# cv.imshow('img', img)
# cv.waitKey()
# cv.destroyAllWindows()