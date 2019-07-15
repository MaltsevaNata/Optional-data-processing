# -*- coding: utf-8 -*-
import os
import numpy as np
#from txttoxml import create_subelement
#from utils_boxes import bbox_transform_l_corner
#from utils_annot import parse_annotation_xml, save_anno_xml
import xml.etree.ElementTree as xml
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


marked_ann = open('/home/user/Документы/for_metrics/gt_val.txt', 'r')
predicted_ann = '/home/user/Документы/for_metrics/hh_vis_det_and_cls_logging'


fail_num = 0

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_objects(file):
    try:
        tree = xml.parse(file)
    except:
        return
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
                #list.append(coords[0] + (coords[2] - coords[0])/2)
                list.append(abs(coords[3]-coords[1]))
                #list.append(coords[1] + (coords[3] - coords[1])/2)
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
        delete = False
        for head in headdress:
            dx = abs(person[1][0] - head[1][0])
            dy = abs(person[1][1] - head[1][1])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
                min_person = person
                min_head = head[0]
                del_head = head
                delete = True
        list = [min_head, min_person[1]]
        pairs.append(list)
        if (delete):
            headdress.remove(del_head)
            delete = False
    return pairs, helmets


def get_color(person):
    return{
            'helmet': 'person_green',
            'head': 'person_red',
            'hat': 'person_red',
            'helmet_off': 'helmet_off',
            'hood': 'person_orange',
            '': 'error'
        }[person[0]]

def get_color_num(person_color):
    return{
            'person_green': 0,
            'person_red': 1,
            'person_orange': 2
        }[person_color]

def compare(pairs, predicted_pairs, TP, FP, FN, y_true, y_pred):
    for pair in pairs:
        for predicted_pair in predicted_pairs:
            pr_name = predicted_pair[0]
            name = pair[0]
            deltax = abs(predicted_pair[1] - pair[1][0])
            deltay = abs(predicted_pair[2] - pair[1][1])
            #if pr_name == 'person_orange' or name == 'person_red' or name == 'person_orange' or pr_name == 'person_red':
               # print('meow')
            if deltax < 220 and deltay < 220:
                y_true.append(name)
                y_pred.append(pr_name)
            if ((pr_name == 'person_orange' and name == 'person_red') or (name == 'person_orange' and pr_name == 'person_red')) and deltax < 220 and deltay < 220:
                FP = FP  -0.5
                #print("red to orange")
            elif pr_name == name and deltax < 220 and deltay < 220:
                TP = TP + 1
            elif pr_name != name and deltax < 220 and deltay < 220:
                FP = FP + 1
            else:
                FN = FN + 1
    return TP, FP, FN, y_true, y_pred



marked = marked_ann.readlines()
print(marked)
predicted = os.listdir(predicted_ann)

#marked.sort()
#predicted.sort()

TP = 0  #недалеко и совпал класс
FP = 0  #класс не совпал
FN = 0  #ничего не совпало

y_true = []
y_pred = []



i = 0
for ann in marked:
    with open(ann.rstrip('\n'), 'r') as f1:
        box1 = get_objects(f1)
        pairs, helmets = get_pairs(box1)
        for person in pairs:
            person[0] = get_color(person)
        for helmet in helmets:
            pairs.append(helmet)
        try:
            f2 = open(predicted_ann + '/' + 'det_' + str(i) + '.xml', 'r')
            predicted_pairs = get_objects(f2)
            TP, FP, FN, y_true, y_pred = compare(pairs, predicted_pairs, TP, FP, FN, y_true, y_pred)
            #y_true, y_pred = fill_pred_arr(pairs, predicted_pairs, y_true, y_pred)
        except:
            print('Failed: ', ann, end='\n ')
            fail_num = fail_num + 1
            i = i + 1
            continue
        i = i + 1

class_names = ['person_green', 'person_red', 'person_orange']
cnf_matrix = confusion_matrix(y_true, y_pred, labels=['person_green', 'person_red', 'person_orange'])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()


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


