# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

tracker = cv2.TrackerCSRT_create()
vs = cv2.VideoCapture("imgs/input-video.mp4")
initBB = None

# loop over frames from the video stream
while vs.isOpened():

    ret,frame = vs.read()

    cv2.line(frame, (933 , 462), (1170 , 462), (0,0,255), 3)
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)

            cX = int((x + x+w) / 2.0)
            cY = int((y + y+h) / 2.0)

            cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

else:
    vs.release()

cv2.destroyAllWindows()