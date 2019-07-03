# -*- coding: utf-8 -*-
import os
import numpy
from utils_boxes import bbox_transform
from utils_annot import parse_annotation_xml, save_anno_xml
import xml.etree.ElementTree as xml


in_ann = '/home/user/Документы/VisDrone2019-DET-train/annotations'
out_ann = '/home/user/Документы/VisDrone2019-DET-train/annotations_xml/'
image_width = 1360
image_height = 765
image_depth = 3
img_end = 'jpg'
folder = 'annotations_xml'


def create_subelement(element, name, text):
    subelement = xml.SubElement(element, name)
    subelement.text = str(text)




annlist = os.listdir(in_ann)
annlist.sort()


for ann in annlist:
    with open(in_ann + '/' + ann, 'r') as f:


        #сначала записываем в xml информацию о фото
        root = xml.Element("annotation")
        fold = xml.Element("folder")
        fold.text = folder
        root.append(fold)
        filename = xml.Element("filename")
        root.append(filename)
        filename.text = ann.replace('.txt', '.jpg')
        path = xml.Element("path")
        path.text = in_ann + '/' + ann
        root.append(path)
        source = xml.Element("source")
        create_subelement(source, "database", "Unknown" )
        root.append(source)
        size = xml.Element("size")
        create_subelement(size, "width", image_width)
        create_subelement(size, "height", image_height)
        create_subelement(size, "depth", image_depth)
        root.append(size)
        segmented = xml.Element("segmented")
        segmented.text = '0'
        root.append(segmented)

        lines = f.readlines()

        for line in lines:
            #каждая строка txt - отдельный объект, создаем из них подэлементы - objects
            obj = line.strip().split(',')

            try:
                box = []
                cls = obj[5]
                class_name = {
                    '0': "ignore",
                    '1': "pedestrian",
                    '2': "people",
                    '3': "bicycle",
                    '4': "car",
                    '5': "van",
                    '6': "truck",
                    '7': "tricycle",
                    '8': "awning-tricycle",
                    '9': "bus",
                    '10': "motor",
                    '11': "others"
                    }.get(cls)

               # van(5), truck(6), tricycle(7), awning - tricycle(8), bus(9), motor(10), others(11)

                #get coordinates
                x = obj[0]
                y = obj[1]
                w = obj[2]
                h = obj[3]

                xmin, ymin, xmax, ymax = bbox_transform([float(x), float(y), float(w), float(h)])
                #get class number

                object = xml.Element("object")
                create_subelement(object, "name", class_name)
                create_subelement(object, "pose", "Unspecified")
                create_subelement(object, "truncated", 0)
                create_subelement(object, "difficult", 0)
                bndbox = xml.SubElement(object, "bndbox")
                create_subelement(bndbox, "xmin", int(xmin))
                create_subelement(bndbox, "ymin", int(ymin))
                create_subelement(bndbox, "xmax", int(xmax))
                create_subelement(bndbox, "ymax", int(ymax))
                #xmi = xml.SubElement(bndbox, "xmin")
                #xmi.text = str(xmin)
                #Возможно, нужно добавить еще подэлементы
                #pose = xml.SubElement(object, "pose")
                #pose.text = "Unspecified"
                root.append(object)



            except:
                continue


    # ‭ ‬создаём новый файл XML с результатами
    tree = xml.ElementTree(root)
    tree.write(out_ann + ann.replace('txt', 'xml'))




