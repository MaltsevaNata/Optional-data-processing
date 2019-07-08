# -*- coding: utf-8 -*-
import os
import numpy
import xml.etree.ElementTree as xml
#import _bpy
from PIL import Image
from utils_boxes import bbox_transform_inv

in_img = 'D:/Study/Python/old_hh_2/2_selected_img'
out_img = 'D:/Study/Python/old_hh_2/cropped_objects'
annotations = 'D:/Study/Python/old_hh_2/2_selected_annot'

def resize(xmin, ymin, xmax, ymax) :
    deltay = int(ymax) - int(ymin)
    deltax = int(xmax) - int(xmin)
    if deltay > deltax:
        xmin = xmin - (deltay - deltax) / 2
        xmax = xmax + (deltay - deltax) / 2
    else:
        ymin = ymin - (deltax - deltay) / 2
        ymax = ymax + (deltax - deltay) / 2
    return xmin, ymin, xmax, ymax

def cut (xmin, ymin, xmax, ymax) :
    deltay = int(ymax) - int(ymin)
    deltax = int(xmax) - int(xmin)
    if deltay < deltax:
        xmin = xmin + (deltax - deltay) / 2
        xmax = xmax - (deltax - deltay) / 2
    else:
        ymin = ymin + (deltay - deltax) / 2
        ymax = ymax - (deltay - deltax) / 2
    return xmin, ymin, xmax, ymax



imglist = os.listdir(in_img)
imglist.sort()

annlist = os.listdir(annotations)
annlist.sort()

for ann in annlist:

    tree = xml.parse(annotations + '/' + ann)
    root = tree.getroot()
    for elem in root:
        for subelem in elem:
            if subelem.tag == 'name':
                classname = subelem.text
            if subelem.tag == 'bndbox':
                box= [None]*4
                i = 0
                for values in subelem:
                    box[i] = int(values.text)
                    i=i+1
                xmin, ymin, xmax, ymax = box

                if (classname == 'helmet' or classname == 'head' or classname == 'helmet_off' or classname == 'hood' or classname == 'hat'):
                    ymin = ymin - 0.1 * ymin
                    ymax = ymax + 0.2 * ymax

                imageObject = Image.open(in_img + '/' + ann.replace('xml', 'jpg'))
                width, height = imageObject.size
                xmin, ymin, xmax, ymax = resize(xmin, ymin, xmax, ymax)
                if xmin < 0 :
                    xmin = 0
                if xmax > width :
                    xmax = width
                if ymin < 0 :
                    ymin =0
                if ymax > height :
                    ymax = height
                box = cut(xmin, ymin, xmax, ymax)
                #box = xmin, ymin, xmax, ymax
                img = in_img + '/' + ann.replace('xml', 'jpg')
                #box = bbox_transform_inv(box)
                # Create an Image object from an Image

                # Crop the image
                cropped = imageObject.crop(box)
                # Display the cropped portion
                save_location = out_img + '/' + classname + '/'+ ann.replace('xml', 'jpg')
                cropped.save(save_location)
                #cropped.show()
