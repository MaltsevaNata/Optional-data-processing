
from main.model.squeezeDet import SqueezeDet
from main.model.dataGenerator import generator_from_data_path, visualization_generator_from_data_path
import keras.backend as K
from keras import optimizers


import numpy as np
import argparse
from keras.utils import multi_gpu_model
from main.config.create_config import load_dict
import cv2
from main.model.evaluation import filter_batch
import math
from PIL import Image
from keras.models import load_model
from utils import parse_annotation_xml, save_anno_xml
import os


#img_name = "n22.jpg"
#img_name = "/home/lab5017/squeezedet2/cubes_datasets/museum1/images_640_480/1_484.jpg"
img_name = "/home/lab5017/squeezedet2/squeezedet-keras_cubes/expirements/broken_net/bb.jpg"
WORKING_DIR = "/home/lab5017/squeezedet2/squeezedet-keras_cubes/expirements/new_and_old_hh/"
save_directory = "images/test_low_thr/"
logging_path = "/home/lab5017/squeezedet2/squeezedet-keras_cubes/images/logging/"

img_file = WORKING_DIR + "img_val.txt"
gt_file = WORKING_DIR + "gt_val.txt"
POST_FILTRATION = True
LOGGING = True
BATCH_SIZE =1

weights = WORKING_DIR + "new_and_old_200eps.hdf5"
CONFIG = WORKING_DIR + "re_hh.config"
color_list = [(255, 255, 255), (255, 10, 10), (0, 0, 255), (0, 255, 0), (10, 10, 220), (10, 230, 250),
              (5, 131, 250)]
#classifier_dict ={0: "background", 1: "hat", 2: "head", 3: "helmet", 4: "helmet_off", 5: "hood"}
#classifier_dict ={0: "background", 1: "hat", 2: "head", 3: "helmet", 4: "helmet_off", 5: "hood", 6: "person"}
classifier_dict = {0: "hat", 1: "head", 2: "helmet", 3: "hood"}
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box

def square(xmin, ymin, xmax, ymax) :
    deltay = int(ymax) - int(ymin)
    deltax = int(xmax) - int(xmin)
    if deltax < 1.4*deltay:
        ymax = ymax - deltay/2
        deltay = int(ymax) - int(ymin)
        deltax = int(xmax) - int(xmin)
    if deltay > deltax:
        xmin = xmin - (deltay - deltax) / 2
        xmax = xmax + (deltay - deltax) / 2
    else:
        ymin = ymin - (deltax - deltay) / 2
        ymax = ymax + (deltax - deltay) / 2
    return xmin, ymin, xmax, ymax

def square_decrease (xmin, ymin, xmax, ymax) :
    deltay = int(ymax) - int(ymin)
    deltax = int(xmax) - int(xmin)
    if deltay < deltax:
        xmin = xmin + (deltax - deltay) / 2
        xmax = xmax - (deltax - deltay) / 2
    else:
        ymin = ymin + (deltay - deltax) / 2
        ymax = ymax - (deltay - deltax) / 2
    return xmin, ymin, xmax, ymax

def prepare_for_classification(img, xmin, ymin, xmax, ymax):
    img_h =img.shape[0]
    img_w =img.shape[1]
    xmin, ymin, xmax, ymax = square(xmin, ymin, xmax, ymax)
    border_touch = 0
    if xmin < 0:
        xmin = 0
        border_touch = 1
    if xmax > img_w:
        xmax = img_w
        border_touch = 1
    if ymin < 0:
        ymin = 0
        border_touch = 1
    if ymax > img_h:
        ymax = img_h
        border_touch = 1
    if border_touch:
        xmin, ymin, xmax, ymax = square_decrease(xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    crop_img = img[ymin:ymax, xmin:xmax]

    x = cv2.resize(crop_img, (224, 224))
    x = x / 255
    x = np.expand_dims(x, axis=0)

    return x

model2 =load_model('/home/lab5017/NIR_Filatov/Keras transfer learning kit light/MN_smcrob_4classes_10eps.h5')

# create config object
cfg = load_dict(CONFIG)

squeeze = SqueezeDet(cfg)
# dummy optimizer for compilation
sgd = optimizers.SGD(lr=cfg.LEARNING_RATE, decay=0, momentum=cfg.MOMENTUM,
                     nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)
squeeze.model.compile(optimizer=sgd,
                      loss=[squeeze.loss], metrics=[squeeze.bbox_loss, squeeze.class_loss,
                                                    squeeze.conf_loss, squeeze.loss_without_regularization])
model = squeeze.model
i = 0

squeeze.model.load_weights(weights)



with open(img_file) as imgs:
    img_names = imgs.read().splitlines()
imgs.close()

count = 0
for img_name in img_names:
    # open img
    img = cv2.imread(img_name).astype(np.float32, copy=False)
    img_copy = cv2.imread(img_name).astype(np.float32, copy=False)
    orig_h =img.shape[0]
    orig_w =img.shape[1]
    # subtract means
    img = (img - np.mean(img)) / np.std(img)
    img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    #cv2.imshow("cropped", img)
    #cv2.waitKey(0)
    img_for_pred = np.expand_dims(img, axis=0)
    print (img_for_pred.shape)


    #squeeze.model.save("mu1_mu2_cv2_smaller_anchors.h5")

    y_pred = model.predict(img_for_pred)  # The first prediction consumes time for initialization of CUDA (0.7-1 seconds)


    print(y_pred.shape)
    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, cfg)

    if POST_FILTRATION:
        # post filrtation of multiple detections of the same object
        pos_thres = 40  # (pixels) threshold which limits distance between the center of good bounding box and fake bbox
        helmet_thres = 220
        conf_thres = 0.00  # threshold which limits minimum difference in confidence to throw  away fake bbox
        # sometimes good bboxes actually have confidence 0.01-0.03 lower than an accurate one

        filtered_boxes = []
        for j, det_box in enumerate(all_filtered_boxes[i]):  # add confidence for ease of comparion
            det_box = np.append(det_box, [float(all_filtered_scores[0][j]), all_filtered_classes[0][j]],
                                axis=0)  # also add classes not to mix them up
            det_box = np.append(det_box, 0) #also append paint flag its color is used inless it's zero
            print(all_filtered_scores[0][j], "- this is scores")
            print(all_filtered_classes[0][j], "- this is classes")

            filtered_boxes.append(det_box)
            print(det_box, "-detbox")

        print("len", len(filtered_boxes))

        for iteration in range(2):  # cycle for  breaks after list.pop action,so for percise result we should go multple times
            print("iteration", iteration)  # thus it is less iterations then to compare everything and then pop elements
            for id1, det_box1 in enumerate(filtered_boxes):
                xc1, yc1, w1, h1, conf1, _, _ = det_box1
                for id2, det_box2 in enumerate(filtered_boxes):
                    xc2, yc2, w2, h2, conf2, _, _ = det_box2
                    dx = abs(xc1 - xc2)
                    dy = abs(yc1 - yc2)
                    r2 = math.sqrt(dx * dx + dy * dy)
                    print(r2)
                    print("id1, 1d2", id1, id2)
                    if r2 < pos_thres:  # if bboxes are suspiciously close
                        if abs(conf1 - conf2) > conf_thres:  # check conf difference
                            if conf1 > conf2:  # so delete lest confident
                                filtered_boxes.pop(id2)
                                print("deletion case1")
                            else:
                                if id1 <= (len(filtered_boxes) - 1):
                                    filtered_boxes.pop(id1)
                                    print("deletion case2")

        classifier_boxes =[]
        #from config: 0 - person, 1 - hat, 2 - helmet, 3 - head, 4 - hood, 5 - helmet_off
        for id1, det_box1 in enumerate(filtered_boxes):
            xc1, yc1, w1, h1, conf1, cls1, _ = det_box1
            if cls1 == 0:
                classifier_boxes.append(det_box1)
                for id2, det_box2 in enumerate(filtered_boxes):
                    xc2, yc2, w2, h2, conf2, cls2 , _ = det_box2
                    xc2, yc2, w2, h2, conf2, _, _ = det_box2
                    dx = abs(xc1 - xc2)
                    dy = abs(yc1 - yc2)
                    r2 = math.sqrt(dx * dx + dy * dy)
                    print(r2)
                    print("id1, 1d2", id1, id2)
                    if r2 < helmet_thres:  # if bboxes are suspiciously close
                        if cls2 == 2:
                            det_box1[6] = 3 #helmet is ok = green, it also has a priority, not to make fake alert
                        elif cls2 == 1 or cls2 == 3 :
                            det_box1[6] = 2 #hat or head state unsafe conditions of work, red from color list
                        elif cls2 == 4:
                            det_box1[6] = 5 #hood, uncertain, orrange
                if det_box1[6] == 0 or det_box1[6] == 2:
                    det_box = bbox_transform_single_box(det_box1[0:4])
                    x_scale = orig_w / cfg.IMAGE_WIDTH
                    y_scale = orig_h / cfg.IMAGE_HEIGHT
                    det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
                    det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)
                    cropped = prepare_for_classification(img_copy, det_box[0], det_box[1], det_box[2], det_box[3])
                    y = model2.predict(cropped, batch_size=1)
                    max1 = int(y.argmax(axis=1))
                    print(img_name, max1)
                    if max1 == 2:
                        det_box1[6] = 3  # helmet is ok = green
                    elif max1 == 0 or max1 == 1:
                        det_box1[6] = 2  # hat or head state unsafe conditions of work, red from color list
                    elif max1 == 3:
                        det_box1[6] = 5  # hood, uncertain, orrange



        # then again devide coordintes, confidences and classes for other finction to use
        only_boxes = [((filtered_boxes[kk])[0:4]) for kk in range(len(filtered_boxes))]
        all_filtered_boxes[i][:] = only_boxes[:]
        only_conf = [((filtered_boxes[kk])[4]) for kk in range(len(filtered_boxes))]
        all_filtered_scores[i][:] = only_conf[:]
        only_classes = [(int((filtered_boxes[kk])[5])) for kk in range(len(filtered_boxes))]
        all_filtered_classes[i][:] = only_classes[:]
        only_paint_flags = [(int((filtered_boxes[kk])[6])) for kk in range(len(filtered_boxes))]
        #all_filtered_classes[i][:] = only_paint_flags[:]
        print(only_paint_flags, "PAINT")

        font = cv2.FONT_HERSHEY_SIMPLEX


        # for box_id,bbox in enumerate(classifier_boxes):
        #     det_box = bbox_transform_single_box(bbox[0:4])
        #     x_scale = orig_w / cfg.IMAGE_WIDTH
        #     y_scale = orig_h / cfg.IMAGE_HEIGHT
        #     det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
        #     det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)
        #     cropped = prepare_for_classification(img_copy, det_box[0], det_box[1], det_box[2], det_box[3] )
        #     x = cv2.resize(cropped, (224, 224))
        #     x = x / 255
        #     x = np.expand_dims(x, axis=0)
        #     y = model2.predict(x, batch_size=1)
        #     max1 = int(y.argmax(axis = 1))
        #     print(max1, "MAX1")
        #     print(y)
        #     new_y = np.delete(y, max1) #bulshit since i don't know which if i shpuld shift or not
        #     max2 = 1+int(new_y.argmax(axis=0))
        #     print(new_y)
        #     print(max2, "MAX2")
        #
        #     cv2.putText(cropped, classifier_dict[max1]+"__"+classifier_dict[max2],
        #                 (30,30), font, 1, (255,255,0), 2, cv2.LINE_AA)
        #     cv2.imwrite("images/crop_test/ct" + str(count) + str(box_id) + ".jpg", cropped)


    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(img_name)


    for j, det_box in enumerate(all_filtered_boxes[0]):
        # transform into xmin, ymin, xmax, ymax
        det_box = bbox_transform_single_box(det_box)
        x_scale = orig_w / cfg.IMAGE_WIDTH
        y_scale = orig_h / cfg.IMAGE_HEIGHT
        det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
        det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)




        #cv2.imshow("cropped", img)
        #cv2.waitKey(0)

        # add rectangle and text
        if only_paint_flags[j]:
            cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), color_list[only_paint_flags[j]], 2)
            cv2.putText(img,
                        cfg.CLASS_NAMES[all_filtered_classes[i][j]] + "" + str('%1.2f' % all_filtered_scores[i][j]),
                        (det_box[0], det_box[1]), font, 1, color_list[only_paint_flags[j]], 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]),
                          color_list[(all_filtered_classes[i][j] + 1)], 2)
            cv2.putText(img, cfg.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str('%1.2f' % all_filtered_scores[i][j]),
                        (det_box[0], det_box[1]), font, 1,
                        color_list[(all_filtered_classes[i][j] + 1)], 1, cv2.LINE_AA)

        print(det_box)

    for j, det_box in enumerate(filtered_boxes):
        # transform into xmin, ymin, xmax, ymax
        det_box[0:4] = bbox_transform_single_box(det_box[0:4])
        x_scale = orig_w / cfg.IMAGE_WIDTH
        y_scale = orig_h / cfg.IMAGE_HEIGHT
        det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
        det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)

    if LOGGING:
        print(img_name)
        xml_boxes = []
        image_depth = 3
        img_end = 'jpg'
        color_person_dict = {3: "person_green", 2: "person_red", 5: "person_orange"}
        for det_box in filtered_boxes:
            if det_box[5] == 0:
                print("key", color_person_dict[det_box[6]])  # for some reason oi need to apply int one more time
                xml_box = [float(det_box[4]), color_person_dict[det_box[6]], int(det_box[0]), int(det_box[1]),
                           int(det_box[2]), int(det_box[3])]
                xml_boxes.append(xml_box)
            if det_box[5] == 5:
                print("det box 5", det_box[5])
                xml_box = [float(det_box[4]), cfg.CLASS_NAMES[int(det_box[5])], int(det_box[0]), int(det_box[1]),
                           int(det_box[2]), int(det_box[3])]
                xml_boxes.append(xml_box)
        if not xml_boxes or (len(xml_boxes) == 1 and det_box[5] == 5):
            for det_box in filtered_boxes:
                if det_box[5] == 1 or det_box[1] == 3:
                    print("wtf_my_dude")
                    xml_box = [float(det_box[4]), color_person_dict[2], int(det_box[0]), int(det_box[1]),
                               int(det_box[2]), int(det_box[3])]
                    print(xml_box)
                    xml_boxes.append(xml_box)
                if det_box[5] == 2:
                    print("wtf_my_dude")
                    xml_box = [float(det_box[4]), color_person_dict[3],  int(det_box[0]), int(det_box[1]),
                               int(det_box[2]), int(det_box[3])]
                    print(xml_box)
                    xml_boxes.append(xml_box)
        save_anno_xml(dir=logging_path,
                      img_name="det_" + str(count),
                      img_format=img_end,
                      img_w=orig_w,
                      img_h=orig_h,
                      img_d=image_depth,
                      boxes=xml_boxes,
                      quiet=False)

    cv2.imwrite("images/hh_vis/pred_vis" + str(count) + ".jpg", img)
    count += 1

