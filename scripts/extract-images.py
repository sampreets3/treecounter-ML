import numpy as np
import cv2

i = 0                                               # Frame counter
frameSkip = 1                                       # Number of frames to be skipped
cap = cv2.VideoCapture("video3.mp4")
fpath = "img/"
logfile = open('video3.txt', "w")           # Store the image names

while(True):
    try:
        ret, frame = cap.read()

    except cv2.error as e:
        pass

    frame = cv2.resize(frame, (800, 600))
    i = i + 1

    if(i%frameSkip == 0):
        cv2.imshow("Output", frame)

        fname = "img-" + str(int(i/frameSkip)) + '.jpg'
        im = cv2.imwrite((fpath + fname), frame)
        logfile.write(fname)
        logfile.write("\n")

        if int(i/frameSkip) == 1:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
print("Total frames : " + str(i))
cv2.destroyAllWindows()
logfile.close()
