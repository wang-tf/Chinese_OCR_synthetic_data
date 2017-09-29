#/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import glob
import random
import codecs
import pathlib
import math
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2
import Aug_Operations as aug


def augmentImage(txt_img, points):
    # Augment rate for eatch type
    rot = random.uniform(0, 1)
    skew_rate = random.uniform(0, 1)
    shear_rate = random.uniform(0, 1)
    distort_rate = random.uniform(0, 1)
    
    if rot < 1:
        rot_degree = random.randint(-10, 10)
        txt_img, points = rotate_img(txt_img, rot_degree)
    elif skew_rate < 0:  # 平行四边形形变
        skew = aug.Skew(1, 'RANDOM', 0.5)
        txt_img, points = skew.perform_operation(txt_img)
    elif shear_rate < 0:  # 剪切形变
        shear= aug.Shear(1., 5, 5)
        txt_img, points = shear.perform_operation(txt_img)
    elif distort_rate < 0:  # 扭曲变形
        distort = aug.Distort(1.0, 4, 4, 1)
        txt_img = distort.perform_operation(txt_img)
    return txt_img, points 


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
    print('Load backgroudn image: %d'%len(image_path_list))
    return image_path_list

def getFontListFromDir(font_dir):
    font_path_list = []
    font_path_list.extend(glob.glob(os.path.join(font_dir, '*.[t,T][t,T][f,F,c,C]')))
    print('Load font files: %d'%len(font_path_list))
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
    return content, content_index

def get_contents(id_character_dict, length_range_tuple, line_number=2):
    contents, contents_index = [], []
    for i in range(line_number):
        content, content_index = get_content(id_character_dict, length_range_tuple)
        contents.append(content)
        contents_index.append(content_index)
    return contents, contents_index

def saveImage2Dir(image, image_save_dir, image_name='test_image'):
    if type(image) == list:
        for index, one_image in enumerate(image):
            saveImage2Dir(one_image, image_save_dir, image_name=image_name+'_'+str(index))
    else:
        image_save_path = os.path.join(image_save_dir, image_name+'.jpg')
        image.save(image_save_path)

def rotate_img(image, degree):
    img = np.array(image)

    height, width = img.shape[:2]

    heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  
    matRotation[1, 2] += (heightNew - height) / 2  
    
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255, 0))

    w = width
    h = height

    points = np.matrix([[-w / 2, -h / 2, 1], [-w / 2, h / 2, 1], [w / 2, h / 2, 1], [w / 2, -h / 2, 1]])

    matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)

    matRotation[0, 2] = widthNew / 2
    matRotation[1, 2] = heightNew / 2

    p = matRotation * points.T

    for row in imgRotation:
        for element in row:
            if element[3] == 0:
                for i in xrange(3):
                    element[i] = 0
    image = Image.fromarray(imgRotation)
    points = np.array(p.T, int)
    return image, points

def mergeImageAtPoint(image, txt_img, left_top_point, color):
    left, top = left_top_point
    image  = pltImage2Array(image)
    
    w, h = txt_img.size
    assert (left+w) <= image.shape[1] and (top+h) <= image.shape[0]  # numpy.shape: height, width, channel
    res_img = np.array(image)

    roi_img = image[top:(top+h), left:(left+w),:]
    if len(color) != 3: color = setColor(roi_img)

    txt_img = np.array(txt_img)
    mask = txt_img[:, :, 0]
    mask1 = mask*1.0/255

    for i in range(0,h-1):
        for j in range(0,w-1):
            res_img[i+top,j+left,:] = color*mask1[i,j] + (1-mask1[i,j])*res_img[i+top,j+left,:]

    res_img = res_img[:,:,[2,1,0]]
    res_img = Image.fromarray(res_img)

    return res_img

def mergeBgimgAndTxtimgPoints(left_center, points):
    left, center_line = left_center
    top = center_line - (max(points[:, 1]) - min(points[:, 1]))/2
    for k in range(0, 4):
        points[k, :] += np.array([left, top])
    return points

def setColor(roi_img):
    if type(roi_img) != np.ndarray:
        roi_img = np.array(roi_img)
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
            draw.line([tuple(point[0]), tuple(point[1]), tuple(point[2]), tuple(point[3]), tuple(point[0])])
            # draw.rectangle((tuple(point[0]), tuple(point[2])))
    del draw
    return image

def getRandomOneFromList(list):
    return list[random.randint(0, len(list)-1)]

def getPointByCenterLine(center_line, left_top_point, width, height):
    left, _ = left_top_point
    points = [(left, center_line - height/2), 
           (left + width, center_line - height/2), 
           (left + width, center_line + height/2), 
           (left, center_line + height/2)]
    return points

def getNewLeftCenterPointByContentPoints(content_points):
    left = content_points[0][0][0]
    bottom_line = getTopOrBottomLineInPoints(content_points, is_top=0)
    top_line = getTopOrBottomLineInPoints(content_points, is_top=1)
    height = bottom_line - top_line
    return (left, bottom_line+height/2)

def getTopOrBottomLineInPoints(points, is_top):
    list = []
    if is_top:
        for i in points:
            list.append(min(i[:, 1]))
        return min(list)
    else:
        for i in points:
            list.append(max(i[:, 1]))
        return max(list)
    
def getPointsOfImageRectangle(width, height):
    return np.array([[0, 0], [width, 0], [width, height], [0, height]])

def cropImageByPoints(image, points,):
    part_image= []
    roi_points_list = []
    for oneline in points:
        roi = getOneLineRectanglePoints(oneline)
        random_roi_tuple = addRandomInROI(roi)
        part_image.append(image.crop(random_roi_tuple))
        roi_points_list.append(roi)
    return part_image, roi_points_list

def addRandomInROI(roi):
    left, top, right, bottom = (roi[0][0],roi[0][1],roi[2][0],roi[2][1])
    max_randint = (bottom - top) / 6
    new_left = random.randint(left-max_randint, left+max_randint/2)
    new_top = random.randint(top-max_randint, top+max_randint/2)
    new_right = random.randint(right-max_randint/2, right+max_randint)
    new_bottom = random.randint(bottom, bottom+2*max_randint)
    return (new_left, new_top, new_right, new_bottom)
    
def getOneLineRectanglePoints(one_line_points):
    top = getTopOrBottomLineInPoints(one_line_points, is_top=1)
    bottom = getTopOrBottomLineInPoints(one_line_points, is_top=0)
    left = min(one_line_points[0][0][0], one_line_points[0][3][0])
    right = max(one_line_points[-1][1][0], one_line_points[-1][2][0])
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
