from hand_tracker import HandDetector
import func as fs
import cv2
import time

detector = HandDetector(maxHands=1)
stime = 0
ctime = 0
k=0
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    hands = detector.findHands(img, draw=False)
    if hands:
        data_img = fs.crop_hand(img, hands, 0)
        cv2.imshow("Hand", data_img)
        k = cv2.waitKey(1)
        if k==ord("s"):
            cv2.imwrite(fr"DATA/ASL/A/IMG{time.time()}.jpg", data_img)
    ctime = time.time()
    fps = 1/(ctime-stime)
    stime = ctime
    cv2.putText(img, f"Fps: {int(fps)}", (10, 30), cv2.FONT_ITALIC, 1, (150, 100, 150), 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)