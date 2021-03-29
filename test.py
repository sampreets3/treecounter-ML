import cv2
import numpy as np 
import argparse

# Argument Parser 
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--input", "-i", default='imgs/input-video.mp4', help="Input Video")
parser.add_argument("--config", "-c", default='models/yolov4/yolov4-custom.cfg', help="YOLO config path")
parser.add_argument("--weights", "-w", default='models/yolov4/yolov4-custom_4000.weights', help="YOLO weights path")
parser.add_argument("--labels", "-l", default='models/yolov4/classes-3.names', help="class names path")
args = parser.parse_args()

cap = cv2.VideoCapture(str(args.input))

# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()