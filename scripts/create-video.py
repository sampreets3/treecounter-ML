import numpy as np
import cv2


i = 0                                               # Frame counter
#cap = cv2.VideoCapture("dataset/videos/video2.mp4")
fpath = "vid/"
logfile = open('log.txt', "w")           # Store the image names

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('test.avi',fourcc, 25.0, (800,600))

while(i < 2001):
	i = i+1
	fname = "img-" + str(i) + '.jpg'
	frame = cv2.imread((fpath + fname))
	
	try:
		cv2.imshow("Output", frame)
		out.write(frame)
		
	except cv2.error as e:
		pass
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
# When everything done, release the capture
out.release()
print("Total frames : " + str(i))
cv2.destroyAllWindows()
logfile.close()
