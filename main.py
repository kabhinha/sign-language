from hand_tracker import HandDetector
from cvzone.ClassificationModule import Classifier
import func as fs
import cv2
import time

detector = HandDetector(maxHands=1)
stime = 0
ctime , c = 0 , 0
lab = ["", "A", "B", "C"]
res, char = '',''
cam = cv2.VideoCapture(0)
cl = Classifier(r"Model\keras_model.h5", r"Model\labels.txt")
while True:
    try:
        s = time.time()
        _, img = cam.read()
        hands, img = detector.findHands(img)
        if hands:
            result_img = fs.crop_hand(img, hands, 0, "Hand")
            pre, ind = cl.getPrediction(result_img, draw=False)
            char = lab[ind]
            if char:
                res+= char
                cv2.putText(img, char, (500, 60), cv2.FONT_ITALIC, 2, (25, 0, 255), 2)
            # cv2.putText(img, str(s-time.time()), (500, 65), cv2.FONT_ITALIC, 1, (255, 0, 255), 1)
            time.sleep(0.5)
        ctime = time.time()
        fps = 1/(ctime-stime)
        stime = ctime
        cv2.putText(img, f"Fps: {int(fps)}", (10, 30), cv2.FONT_ITALIC, 1, (150, 100, 150), 1)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k==ord("s"):
            break
    except Exception:
        pass
print(res)