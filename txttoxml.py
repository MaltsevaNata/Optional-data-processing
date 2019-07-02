import os
import numpy

in_ann =  ' D:/Study/Neuro/VisDrone/VisDrone2019-DET-val/annotations/ '
out_ann = r' D:/Study/Neuro/VisDrone/VisDrone2019-DET-val/annotations_xml/ '

def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

    return out_box


with open(in_ann, 'r') as f:
    lines = f.readlines()
f.close()
for line in lines:

    obj = line.strip().split(',')

    #get class, if class is not in listed, skip it
    try:
        #cls = config.CLASS_TO_IDX[obj[6].lower().strip()]
        #print(cls)
        #print("fffff")


        #get coordinates
        x = float(obj[0])
        y = float(obj[1])
        w = float(obj[3])
        h = float(obj[3])
    except:
        continue