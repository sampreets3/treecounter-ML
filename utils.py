import numpy as np
import cv2

def get_centroid(x1, y1, x2, y2):
    return (int((x1+x2)/2) ,int((y1 + y2)/2))

def is_lower_than_line(l_start_x, l_start_y, l_end_x, l_end_y, pt_x, pt_y):
    d = ((pt_x - l_start_x)*(l_end_y - l_start_y)) - ((pt_y - l_start_y)*(l_end_x - l_start_x))
    if d < 0:
        return 0

    elif d == 0:
        return 1

    else :
        return 2

def draw_bboxes(img, bboxes, confidences, classes):
    """
    Draw the bounding boxes on the image

    Parameters:
    -----------
    img : numpy.ndarray
        Image or video frame.
    bboxes : numpy.ndarray
        Bounding Boxes pixel coordinates.

    Returns:
    --------
    image : numpy.ndarray
        Updated Image

    """
    CONF_THRESH, NMS_THRESH = 0.5, 0.5

    # Set display colours for each class
    colors = [(255, 128, 0), (0, 128, 255), (0, 255, 128)]

    # Perform NMS on the bboxes
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
    # Get class labels
    with open(classes, "r") as f:
        classes_id = [line.strip() for line in f.readlines()]

    for index in indices:
        print(index)
        x, y, w, h = bboxes[index]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, classes_id[index], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    return img