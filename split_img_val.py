import random
import numpy as np
import argparse
from PIL import Image

classes = ['hat', 'head', 'helmet', 'hood']

def split():

    for item in classes:
        img_file = "/home/user/Документы/small_cropped_objects_2/" + item + ".txt"

        percent_train = 70
        percent_val = 20
        percent_test = 10

        print(percent_test)


        if not (percent_train + percent_test + percent_val) == 100:
            print("Percentages have to sum to 100")

        with open(img_file) as imgs:
            img_names = imgs.read().splitlines()
        imgs.close()

        shuffle = 1

        if shuffle:
            # permutate images
            shuffled = list(img_names)
            random.shuffle(shuffled)
            img_names = shuffled

        n_train = int(np.floor(len(img_names) * percent_train / 100))

        n_val = int(np.floor(len(img_names) * (percent_train + percent_val) / 100))

        #assert len(img_names) == len(gt_names)


        with open("/home/user/Keras transfer learning kit light/hard_hat_2/" + item +"/img_train.txt", 'w') as img_train:
            train = "\n".join(img_names[0:n_train])
            img_train.write(train)

        # img_train.close()


        with open("/home/user/Keras transfer learning kit light/hard_hat_2/" + item +"/img_val.txt", 'w') as img_val:
            val = "\n".join(img_names[n_train:n_val])
            img_val.write(val)

        # img_train.close()


        with open("/home/user/Keras transfer learning kit light/hard_hat_2/" + item +"/img_test.txt", 'w') as img_test:
            test = "\n".join(img_names[n_val:])
            img_test.write(test)

        # img_train.close()



    print("Imgs splitted")

    # gt_val.close()

def get_img() :
    i = 1
    text_files = ['test', 'train', 'val']
    for item in classes:
        i = i + 1000
        for text_file in text_files:
            img_file = '/home/user/Keras transfer learning kit light/hard_hat_2/' + item + '/img_'+ text_file + '.txt'
            save_location = '/home/user/Keras transfer learning kit light/hard_hat_2/select_without_person/' + text_file + '/' + item + '/'
            with open(img_file) as imgs:
                img_names = imgs.read().splitlines()

                for img in img_names:
                    try:
                        imageObject = Image.open(img)
                        imageObject.save(save_location + str(i) + '.jpg')
                        i = i+1
                    except:
                        continue

get_img()