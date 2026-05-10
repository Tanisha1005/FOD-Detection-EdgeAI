import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not read")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()