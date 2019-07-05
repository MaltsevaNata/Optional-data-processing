# -*- coding: utf-8 -*-
import os
import numpy
#from txttoxml import create_subelement
from utils_boxes import bbox_transform_l_corner
from utils_annot import parse_annotation_xml, save_anno_xml
import xml.etree.ElementTree as xml

input_xml = '/home/user/Документы/old_hard_hat/old_hh/File_1.xml'
folder = '/home/user/Документы/old_hard_hat/old_hh/annotations_xml/'
image_width = 1600
image_height = 1200
image_depth = 3

def create_subelement(element, name, text):
    subelement = xml.SubElement(element, name)
    subelement.text = str(text)

def create_object(pictures, class_name):
    for objects in pictures.iter('ObjectRect'):
        object = xml.Element("object")
        create_subelement(object, "name", class_name)
        create_subelement(object, "pose", "Unspecified")
        create_subelement(object, "truncated", 0)
        create_subelement(object, "difficult", 0)

        bndbox = xml.SubElement(object, "bndbox")

        koord = objects.text
        koord = koord.strip().split()
        koord = map(int, koord)
        xmin, ymin, xmax, ymax = bbox_transform_l_corner(koord)

        create_subelement(bndbox, "xmin", int(xmin))
        create_subelement(bndbox, "ymin", int(ymin))
        create_subelement(bndbox, "xmax", int(xmax))
        create_subelement(bndbox, "ymax", int(ymax))
        root.append(object)



old_tree = xml.parse(input_xml)
old_root = old_tree.getroot()
for elem in old_root:
    created_files = []
    for subelem in elem:
        class_name = subelem.find('Label').text
        PictureInfo = subelem.find('PictureInfo')
        for pictures in PictureInfo:
            exitFlag = False
            name = pictures[0].text
            #if name == '"01228.jpg"':
            for file in created_files:
                if name == file:
                    #Значит файл уже создан, нужно добавить в него еще объект
                    xml_name = name.replace('jpg', 'xml').strip('""')
                    tree = xml.parse(folder + xml_name)
                    root = tree.getroot()
                    create_object(pictures, class_name)
                    tree.write(folder + xml_name)
                    exitFlag = True
                    break

            if not exitFlag:
                #создаем новый файл
                # Каждый блок <_> содержит информацию об отдельном фото
                #создаем отдельный файл под каждый блок
                root = xml.Element("annotation")

                fold = xml.Element("folder")
                fold.text = folder
                root.append(fold)

                filename = xml.Element("filename")
                root.append(filename)
                filename.text = name

                path = xml.Element("path")
                path.text = folder + filename.text
                root.append(path)

                source = xml.Element("source")
                create_subelement(source, "database", "Unknown")
                root.append(source)

                size = xml.Element("size")
                create_subelement(size, "width", image_width)
                create_subelement(size, "height", image_height)
                create_subelement(size, "depth", image_depth)
                root.append(size)

                segmented = xml.Element("segmented")
                segmented.text = '0'
                root.append(segmented)


                try:
                    create_object(pictures, class_name)
                except:
                    continue

                # ‭ ‬создаём новый файл XML
                tree = xml.ElementTree(root)
                tree.write(folder + filename.text.replace('jpg', 'xml').strip('""'))
                created_files.append(filename.text)
