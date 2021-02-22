def get_centroid(x1, y1, x2, y2):
    return int((x1+x2)/2) ,int((y1 + y2)/2)

def is_lower_than_line(l_start_x, l_start_y, l_end_x, l_end_y, pt_x, pt_y):
    d = ((pt_x - l_start_x)*(l_end_y - l_start_y)) - ((pt_y - l_start_y)*(l_end_x - l_start_x))
    if d < 0:
        return 0

    elif d == 0:
        return 1

    else :
        return 2
