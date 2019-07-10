# -*- coding: utf-8 -*-
import os
import numpy
#from txttoxml import create_subelement
from utils_boxes import bbox_transform_l_corner
from utils_annot import parse_annotation_xml, save_anno_xml
import xml.etree.ElementTree as xml
import math


marked_ann = 'D:/Study/Python/old_hh/check_annot'
predicted_ann = 'D:/Study/Python/old_hh/predicted_annot'


def get_objects(file):
    tree = xml.parse(file)
    root = tree.getroot()
    objects = [[None]*3 for i in (range(len(root)-6))] # 0- classname, 1,2 - center coordinates(x,y)
    i = 0
    for elem in root:
        for subelem in elem:
            if subelem.tag == 'name':
                objects[i][0] = subelem.text

            if subelem.tag == 'bndbox':
                coords=[]
                for values in subelem:
                    coords.append(int(values.text))
                objects[i][1] = abs(coords[2]-coords[0]) #xmax-xmin
                objects[i][2] = abs(coords[3]-coords[2]) #ymax - ymin
                i = i + 1

    return objects


def get_pairs(objects):
    people = []
    head_types = ['helmet' , 'head', 'hat', 'hood']
    headdress = []
    for obj in range(len(objects)):
        list=[]
        list.append( objects[obj][0])
        list.append(objects[obj][1:3])#center of the object
        for type in head_types:
            if objects[obj][0] == type:
                headdress.append(list)
                continue
        if objects[obj][0] == 'person':
                people.append(list)
    pairs = []
    for person in people:
        min_dist = 20000
        min_head = ''
        for head in headdress:
            dx = abs(person[1][0] - head[1][0])
            dy = abs(person[1][1] - head[1][1])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
                min_person = person
                min_head = head[0]
                delete = head
        list = [min_head, min_person[1]]
        pairs.append(list)
        headdress.remove(delete)
    return pairs


def get_color(person):
    return{
            'helmet': 'person_green',
            'helmet_off' or 'head' or 'hat': 'person_red',
            'hood': 'person_orange'
        }[person[0]]


marked = os.listdir(marked_ann)
predicted = os.listdir(predicted_ann)

marked.sort()
predicted.sort()

for ann in marked:
    with open(marked_ann + '/' + ann, 'r') as f1:
        box1 = get_objects(f1)
        pairs = get_pairs(box1)
        for person in pairs:
            person[0] = get_color(person)
        f2 = open(predicted_ann + '/' + ann, 'r')
        #box2 = get_objects(f2) ?????





