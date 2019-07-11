# -*- coding: utf-8 -*-
import os
import numpy
#from txttoxml import create_subelement
from utils_boxes import bbox_transform_l_corner
from utils_annot import parse_annotation_xml, save_anno_xml
import xml.etree.ElementTree as xml
import math


marked_ann = open('/home/user/Документы/for_metrics/gt_val.txt', 'r')
predicted_ann = '/home/user/Документы/for_metrics/hh_vis_det_and_cls_logging'


def get_objects(file):
    tree = xml.parse(file)
    root = tree.getroot()
    objects = []
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
                list.append(abs(coords[2]-coords[0]))
                list.append(abs(coords[3]-coords[2]))
                objects.append(list)
    return objects


def get_pairs(objects):
    people = []
    head_types = ['helmet', 'head', 'hat', 'hood']
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
    if (len(people) != 0) and (len(headdress) == 0):
        return pairs, helmets
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

def compare(pairs, predicted_pairs, TP, FP, FN):
    guessed = 0
    not_guessed = len(pairs)
    for pair in pairs:
        for predicted_pair in predicted_pairs:
            deltax = predicted_pair[1] - pair[1][0]
            deltay = predicted_pair[2] - pair[1][1]
            if pr_name == name and deltax < 220 and deltay < 220:
                TP = TP + 1
            elif pr_name != name and deltax < 220 and deltay < 220:
                FP = FP + 1
            else:
                FN = FN + 1
    return TP, FP, FN


marked = marked_ann.readlines()
predicted = os.listdir(predicted_ann)

marked.sort()
predicted.sort()

TP = 0  #недалеко и совпал класс
FP = 0  #класс не совпал
FN = 0  #ничего не совпало
fail_num = 0

for ann in marked:
    try:
        i = 0
        with open(ann[:-1], 'r') as f1:
            box1 = get_objects(f1)
            pairs, helmets = get_pairs(box1)
            for person in pairs:
                person[0] = get_color(person)
            for helmet in helmets:
                pairs.append(helmet)
            f2 = open(predicted_ann + '/' + 'det_' + str(i) + '.xml', 'r')
            predicted_pairs = get_objects(f2)
            TP, FP, FN = compare(pairs, predicted_pairs, TP, FP, FN)
    except:
        print('Failed: ', ann, end='\n ')
        fail_num = fail_num + 1
        continue
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2 * (precision * recall)/(precision + recall)
print('\n Failed ', fail_num, end=' images \n ')
print('\n TP: ', TP, end=' \n ')
print('\n FP: ', FP, end=' \n ')
print('\n FN: ', FN, end=' \n ')
print('\n precision: ', precision, end=' \n ')
print('\n recall: ', recall, end=' \n ')
print('\n F1: ', F1, end=' \n ')


