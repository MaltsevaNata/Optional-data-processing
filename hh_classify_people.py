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
    #objects = [[None]*3 for i in (range(len(root)-6))] # 0- classname, 1,2 - center coordinates(x,y)
    objects = []
    #i = 0
    for elem in root:
        list = []
        for subelem in elem:

            if subelem.tag == 'name':
                list.append(subelem.text)
               # objects[i][0] = subelem.text

            if subelem.tag == 'bndbox':
                coords=[]
                for values in subelem:
                    coords.append(int(values.text))
                #objects[i][1] = abs(coords[2]-coords[0]) #xmax-xmin
                list.append(abs(coords[2]-coords[0]))
                #objects[i][2] = abs(coords[3]-coords[2]) #ymax - ymin
                list.append(abs(coords[3]-coords[2]))
                objects.append(list)
                #i = i + 1

    return objects


def get_pairs(objects):
    people = []
    head_types = ['helmet' , 'head', 'hat', 'hood']
    headdress = []
    helmets = []
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
        if objects[obj][0] == 'helmet_off':
            helmets.append(list)
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
    return pairs, helmets


def get_color(person):
    return{
            'helmet': 'person_green',
            'head': 'person_red',
            'hat': 'person_red',
            'helmet_off': 'helmet_off',
            'hood': 'person_orange'
        }[person[0]]

def compare(pairs, predicted_pairs):
    guessed = 0
    not_guessed = len(pairs)
    for pair in pairs:
        for predicted_pair in predicted_pairs:
            pr_name = predicted_pair[0]
            name = pair[0]
            pr_x = predicted_pair[1]
            pr_y = predicted_pair[2]
            x = pair[1][0]
            y = pair[1][1]
            deltax = predicted_pair[1] - pair[1][0]
            deltay = predicted_pair[2] - pair[1][1]
            if pr_name == name and deltax < 200 and deltay < 200:
                guessed = guessed + 1
                not_guessed = not_guessed - 1
    return guessed, not_guessed


marked = os.listdir(marked_ann)
predicted = os.listdir(predicted_ann)

marked.sort()
predicted.sort()

for ann in marked:
    with open(marked_ann + '/' + ann, 'r') as f1:
        box1 = get_objects(f1)
        pairs, helmets = get_pairs(box1)
        for person in pairs:
            person[0] = get_color(person)
        for helmet in helmets:
            pairs.append(helmet)
        f2 = open(predicted_ann + '/' + ann, 'r')
        predicted_pairs = get_objects(f2)
        guessed, not_guessed = compare(pairs, predicted_pairs)
        print('Guessed: ', guessed, end='\n')
        print('Not guessed: ', not_guessed, end='')




