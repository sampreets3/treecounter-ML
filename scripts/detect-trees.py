import numpy as np
import argparse as ap
import utils
import cv2


#Configure the argument parser----------------------------
parser = ap.ArgumentParser(description="Detects and counts the number of trees in a video")

parser.add_argument("--input", "-iv", required=True, help="The location of the input video file")
parser.add_argument("--output", "-ov", required=True, help="The location of the output video file")
parser.add_argument("--config", "-cfg", required=False, help="The model configuration file")
parser.add_argument("--weights", "-w", required=False, help="The model weights file")
parser.add_argument("--classes", "-c", required=False, help="The file containing the class descriptions")
args = parser.parse_args()

# Default usage
"""
python3 scripts/detect-trees.py \
-iv videos/video3.mp4  \
-ov videos/test.avi \
-cfg models/yolov3/cfg/yolov3_custom.cfg \
-w models/yolov3/weights/yolov3_custom_final.weights \
-c models/yolov3/classes.names
"""
#---------------------------------------------------------

# set the VideoCapture and VideoWriter objects------------
cap = cv2.VideoCapture(str(args.input))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(str(args.output),fourcc, 25.0, (800,600))
#---------------------------------------------------------

#Parameters for the darknet-------------------------------
config = str(args.config)
weights = str(args.weights)
names = str(args.classes)
CONF_THRESH, NMS_THRESH = 0.1, 0.1
#---------------------------------------------------------

# Skip frames for lower latency---------------------------
i = 0
frameSkip = 25
#---------------------------------------------------------

# LOS configurations--------------------------------------
left_line_start = (70, 500)
left_line_end   = (230, 500)

right_line_start = (550, 500)
right_line_end   = (750, 500)

line_color = (255, 0, 0)
line_thickness = 2
#---------------------------------------------------------

# Load the network----------------------------------------
#net = cv2.dnn.readNetFromDarknet(config, weights)
net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#---------------------------------------------------------

while(True):
    i = i + 1
    if (i % frameSkip == 0):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = cv2.resize(frame,(800, 600))
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width= img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        class_ids, confidences, b_boxes = [], [], []
        centroids = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width,    height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    b_boxes.append([x, y, int(w), int(h)])
                    centroid_x, centroid_y = utils.get_centroid(x, y, x+int(w), y+int(h))
                    centroids.append([centroid_x, centroid_y])
                    # Add a horizontal line, and count


                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

                    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident    bounding boxes
                    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

                    # Draw the filtered bounding boxes with their class to the image
                    with open(names, "r") as f:
                        classes = [line.strip() for line in f.readlines()]
                        colors = np.random.uniform(0, 255, size=(len(classes), 3))

                        for index in indices:
                            x, y, w, h = b_boxes[index]
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                            cv2.putText(img, classes[class_ids[index]], (x - 10, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                            for c in centroids:
                                c_x, c_y = c[0], c[1]
                                cv2.circle(img, (c_x, c_y), radius=2, color=(0, 255, 255), thickness=2)
                                cv2.line(img, left_line_start, left_line_end, line_color, line_thickness)
                                cv2.line(img, right_line_start, right_line_end, line_color, line_thickness)
                                cv2.putText(img, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 120, 68), 2)

        # Display the resulting frame
        cv2.imshow('image',img)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
