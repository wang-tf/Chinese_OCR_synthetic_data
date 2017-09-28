# -*- coding:utf-8 -*-

import os
import glob
import random
import codecs
import pathlib
import math
from PIL import Image, ImageDraw
import numpy as np
import cv2
import Aug_Operations as aug

def getTopNCharacters2Dict(characters_file_path, top_n):
    id_cha_dict = {}
    characters = getAllCharactersFromFile(characters_file_path)
    for index, key in enumerate(characters):
        id_cha_dict[index] = key
        if index == top_n-1: break
    return id_cha_dict

def getAllCharactersFromFile(characters_file_path):
    characters_set = set()
    with codecs.open(characters_file_path, encoding="utf-8") as file:
        lines = file.readlines()
        print('lines')
        for oneline in lines:
                characters_set.update(list(oneline.replace(' ','')))
    return list(characters_set)

def makeDirectory(path):
    if os.path.exists(path):
        print('The path exists: %s'%path)
    else:
        os.mkdir(path)
        print('The path maked: %s'%path)

def getBackgroundListFromDir(background_dir):
    image_path_list = []
    image_path_list.extend([path for path in pathlib.Path(background_dir).rglob('*.jpg')])
    return image_path_list

def getFontListFromDir(font_dir):
    font_path_list = []
    font_path_list.extend(glob.glob(os.path.join(font_dir, '*.[t,T][t,T][f,F,c,C]')))
    return font_path_list

def get_content(id_character_dict, length_range_tuple):
    length = len(id_character_dict)
    rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
    content = u''
    content_index = []
    for i in range(rand_len):
        rand_index = random.randint(0, length-1)
        content += id_character_dict[rand_index]
        content_index.append(rand_index)
    print(content)
    return content, content_index

def get_contents(id_character_dict, length_range_tuple, line_number=2):
    contents, contents_index = [], []
    for i in range(line_number):
        content, content_index = get_content(id_character_dict, length_range_tuple)
        contents.append(content)
        contents_index.append(content_index)
    return contents, contents_index

def saveImage2Dir(image, image_save_dir, image_name='test_image'):
    image_save_path = os.path.join(image_save_dir, image_name+'.jpg')
    image.save(image_save_path)


def augmentImage(txt_img):
    # Augment rate for eatch type
    rot = random.uniform(0, 1)
    skew_rate = random.uniform(0, 1)
    shear_rate = random.uniform(0, 1)
    distort_rate = random.uniform(0, 1)
    
    if rot < 0:  # 旋转
        rot_degree = random.randint(-10, 10)
        txt_img, points = rotate_img(txt_img, rot_degree)
    elif skew_rate < 1:  # 平行四边形形变
        skew = aug.Skew(1, 'RANDOM', 0.5)
        txt_img, points = skew.perform_operation(txt_img)
    elif shear_rate < 1:  # 剪切形变
        shear= aug.Shear(1., 5, 5)
        txt_img, points = shear.perform_operation(txt_img)
    elif distort_rate < 1:  # 扭曲变形
        distort = aug.Distort(1.0, 4, 4, 1)
        txt_img = distort.perform_operation(txt_img)
    return txt_img, points 

# 旋转图像并保留原图像的全部区域
def rotate_img(image, degree):

    img = np.array(image)

    height, width = img.shape[:2]

    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

    w = width
    h = height

    points = np.matrix([[-w / 2, -h / 2, 1], [-w / 2, h / 2, 1], [w / 2, h / 2, 1], [w / 2, -h / 2, 1]])

    matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)

    matRotation[0, 2] = widthNew / 2
    matRotation[1, 2] = heightNew / 2

    p = matRotation * points.T

    #cv2.imshow("img", img)
    #cv2.imshow("imgRotation", imgRotation)
    #cv2.waitKey(0)

    image = Image.fromarray(imgRotation)

    return image, p.T

def mergeImageAtPoint(image, txt_img, left_top_point):
    left, top = left_top_point
    image  = pltImage2Array(image)
    
    w, h = txt_img.size
    assert (left+w) <= image.shape[1] and (top+h) <= image.shape[0]  # numpy.shape: height, width, channel
    res_img = np.array(image)

    roi_img = image[top:(top+h), left:(left+w),:]
    color = setColor(roi_img)

    txt_img = np.array(txt_img)
    mask = txt_img[:, :, 0]
    mask1 = mask*1.0/255

    for i in range(0,h-1):
        for j in range(0,w-1):
            res_img[i+top,j+left,:] = color*mask1[i,j] + (1-mask1[i,j])*res_img[i+top,j+left,:]

    res_img = res_img[:,:,[2,1,0]]
    res_img = Image.fromarray(res_img)

    return res_img

def mergeBgimgAndTxtimgPoints(left_top_point, points):
    left, top = left_top_point
    for k in range(0, 4):
        points[k, :] += np.array([left, top])
    return points

def setColor(roi_img):
    color1 = cv2.mean(roi_img)
    color = np.zeros(3, dtype=np.int)
    color[0] = math.ceil(color1[2])
    color[2] = math.ceil(color1[1])
    color[1] = math.ceil(color1[0])
    for k in range(0,3):
        s = random.randint(0, 2)
        if color[s] > 150:
            color[s] = random.randint(0,20)
        else:
            color[s] = random.randint(230,255)
    return color

def pltImage2Array(image):
    image = image.convert('RGB')
    image = np.array(image) 
    # Convert RGB to BGR 
    image = image[:, :, ::-1].copy() 
    return image

def saveIdCharacterDict2File(id_character_dict, save_path):
    with codecs.open(save_path, 'w', encoding='utf-8') as file:
        for (key, val) in id_character_dict.items():
            file.write(str(key) + ' ' + val + '\n')  # the first index must be zero
    return

def drawMulContentsRectangle(image, mulcontents_points):
    draw = ImageDraw.Draw(image)
    for content_points in mulcontents_points:
        for point in content_points:
            draw.rectangle((tuple(point[0]), tuple(point[2])))
    del draw
    return image

def getRandomOneFromList(list):
    return list[random.randint(0, len(list)-1)]