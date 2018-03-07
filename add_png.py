import cv2
import os
import random
import glob
import numpy as np
import xml.etree.ElementTree as ET
from datasets.tea_dataset import *

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    channels = img.shape[2]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
    return img


def rotateImg(img, angle):
    height, width = img.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def get_correct_pos(X, Y, H, W, h, w, max_try=3e+3):
    i=0
    x = random.randint(0,W)
    y = random.randint(0,H)
    while True:
        freeX = set(range(50, W-50)).symmetric_difference(X)
        freeY = set(range(50, H-50)).symmetric_difference(Y)
        freeX = freeX - freeX.intersection(X)
        freeY = freeY - freeY.intersection(Y)
        if set(range(x, x+w)).intersection(X) or x+w>=W-50 or \
                y+h>=H-50 or set(range(y, y + h)).intersection(Y):
            if len(freeY) > 0 and len(freeX) > 0:
                x = random.sample(freeX, 1)[0]
                y = random.sample(freeY, 1)[0]
            else:
                continue
            i += 1
        else:
            break
        if i==max_try:
            return 0, 0, X, Y
            break
    X.extend(range(x, x+w))
    Y.extend(range(y, y+h))
    return x, y, X, Y


def get_positions(annot_file):
    tree = ET.parse(annot_file)
    X = []
    Y = []
    for obj in tree.findall('object'):
        if obj.find('deleted') == True and int(obj.find('deleted').text) == 1:
            continue
        else:
            bbox = obj.find('bndbox')
            if bbox.find('xmin').text.isdigit()==True:
                x1 = int(bbox.find('xmin').text)
            if bbox.find('ymin').text.isdigit()==True:
                y1 = int(bbox.find('ymin').text)
            if bbox.find('xmax').text.isdigit()==True:
                x2 = int(bbox.find('xmax').text)
            if bbox.find('ymax').text.isdigit()==True:
                y2 = int(bbox.find('ymax').text)
            X.extend(range(x1, x2))
            Y.extend(range(y1, y2))
    return X,Y

def paste_obj_color(image_file, annot_file, input_shape, box_annots, max=7):
    imgs_small = [glob.glob("/data/Tmp_datasets/tea_img/*.png")]
    img = cv2.imread(image_file, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    annot_file = (os.path.join(box_annots, annot_file))
    X, Y = get_positions(annot_file)
    H, W = img.shape[:2]
    random.shuffle(imgs_small[0])
    for j in range(max):
        img_small = cv2.imread(imgs_small[0][j], cv2.IMREAD_UNCHANGED)
        img_small = rotateImg(img_small, random.randint(0, 90))
        h, w = img_small.shape[:2]
        x, y, X, Y = get_correct_pos(X, Y, H, W, h, w)
        if x != 0:
            img = overlay_image_alpha(img,
                                      img_small[:, :, 0:3],
                                      (x, y),
                                      img_small[:, :, 3] / 255.0)
        else:
            pass

    img = cv2.resize(img, dsize=(input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)

    return img


def paste_obj_bw(image_file, annot_file, input_shape, box_annots, max=7, rescale=1. / 255):
    imgs_small = [glob.glob("/data/Tmp_datasets/tea_img/*.png")]
    img = cv2.imread(image_file, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    annot_file = (os.path.join(box_annots, annot_file))
    X, Y = get_positions(annot_file)
    H, W = img.shape[:2]
    random.shuffle(imgs_small[0])
    for j in range(max):
        img_small = cv2.imread(imgs_small[0][j], cv2.IMREAD_UNCHANGED)
        img_small = rotateImg(img_small, random.randint(0, 90))
        h, w = img_small.shape[:2]
        x, y, X, Y = get_correct_pos(X, Y, H, W, h, w)
        if x != 0:
            img = overlay_image_alpha(img,
                                      img_small[:, :, 0:3],
                                      (x, y),
                                      img_small[:, :, 3] / 255.0)
        else:
            pass

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img = np.zeros(img.shape)
    out_img[:, :, 0] = img_gray
    out_img[:, :, 1] = img_gray
    out_img[:, :, 2] = img_gray
    out_img = cv2.resize(out_img, (input_shape[1], input_shape[0]))
    img = out_img * rescale

    return img


def main():
    imgs_small = [glob.glob("/data/Tmp_datasets/tea_img/*.png")]
    filenames = read_sets_file(tr_imagesets_folder, "train")

    img_large = cv2.imread(os.path.join(tr_image_folder, filenames[1] + ".jpg"),
                            cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    annot_file = (os.path.join(box_annotation_folder, filenames[1] + ".xml"))
    X,Y = get_positions(annot_file)
    max = 5
    img = img_large
    H, W = img.shape[:2]
    random.shuffle(imgs_small[0])
    for j in range(max):
        img_small = cv2.imread(imgs_small[0][j], cv2.IMREAD_UNCHANGED)
        print(imgs_small[0][j])
        img_small = rotateImg(img_small, random.randint(0, 90))
        h, w = img_small.shape[:2]
        x, y, X, Y = get_correct_pos(X, Y, H, W, h, w)
        if x != 0:
            img = overlay_image_alpha(img,
                                  img_small[:, :, 0:3],
                                  (x, y),
                                  img_small[:, :, 3] / 255.0)
        else:
            pass

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('img',img)
    cv2.waitKey(0)

if  __name__ =='__main__':main()
