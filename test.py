import cv2
import numpy as np 
import argparse
import utils
import sort

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

CONF_THRESH, NMS_THRESH = 0.5, 0.5
count = 0

# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

track = cv2.TrackerCSRT_create()

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    height, width = frame.shape[:2]

    if frame is None:
        break
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    cv2.line(frame, (0, 450), (800, 450), (255, 50, 25), 2, cv2.LINE_AA)
    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Initialise the tracker with bbox
            #x,y,w,h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
            #bbox = frame[x:x+w, y:y+h]
            #track.init(frame, bbox)
            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        #(success, box) = track.update(frame, b_boxes)

        #if success:
        #    (x, y, w, h) = [int(v) for v in box]
        #    cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
    #print(indices)

    with open(args.labels, "r") as f:
       classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for index in indices:
    #    print(index)
       x, y, w, h = b_boxes[index]
       # calculate centroid
       (centroid_x, centroid_y) = utils.get_centroid(x, y, x+w, y+h)

       cv2.circle(frame, (centroid_x, centroid_y), 2, (125, 125, 0), 2)
       if (centroid_y == 450) and classes[class_ids[index]] == "tree":
           count += 1
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
       text_label = "{}".format(classes[class_ids[index]])
       (label_width, label_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
       y_label = max(y, label_height)
       cv2.rectangle(frame, (x, y_label - label_height), (x + label_width, y_label + baseline), (0, 0, 0), cv2.FILLED)
       cv2.putText(frame, classes[class_ids[index]], (x , y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 255, 255), 1)

       text_count = "Trees: {}".format(count)
       cv2.putText(frame, text_count, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
    
    #Display the frame
    cv2.imshow("Frame", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()