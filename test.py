import cv2
import numpy as np 
import argparse
import utils
#import sort

# Argument Parser 
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--input", "-i", default='imgs/input-video.mp4', help="Input Video")
parser.add_argument("--output", "-o", default="imgs/result.mp4", help="Output video")
parser.add_argument("--config", "-c", default='models/yolov4/yolov4-custom.cfg', help="YOLO config path")
parser.add_argument("--weights", "-w", default='models/yolov4/yolov4-custom_4000.weights', help="YOLO weights path")
parser.add_argument("--labels", "-l", default='models/yolov4/classes-3.names', help="class names path")
args = parser.parse_args()

cap = cv2.VideoCapture(str(args.input))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(args.output), fourcc, 25.0, (800, 600))

CONF_THRESH, NMS_THRESH = 0.75, 0.5
count = 0

# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    # Read frame from video
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    height, width = frame.shape[:2]

    # Handle end of video exception
    if frame is None:
        break
    
    # Construct blob from image
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Do a forward pass with the blob
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

    with open(args.labels, "r") as f:
       classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for index in indices:
       x, y, w, h = b_boxes[index]

       # calculate centroid
       (centroid_x, centroid_y) = utils.get_centroid(x, y, x+w, y+h)
       cv2.circle(frame, (centroid_x, centroid_y), 2, (125, 125, 0), 2)

       # Display the bounding boxes    
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)               # Bounding box
       text_label = "{}".format(classes[class_ids[index]])
       (label_width, label_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
       y_label = max(y, label_height)
       cv2.rectangle(frame, (x, y_label - label_height), (x + label_width, y_label + baseline), (0, 0, 0), cv2.FILLED) # Label box
       cv2.putText(frame, classes[class_ids[index]], (x , y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 255, 255), 1) # Label text
    
    #Display the frame
    cv2.imshow("Frame", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()