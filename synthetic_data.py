#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
from PIL import Image, ImageDraw, ImageFont
import utils 
import codecs
import tqdm


def main():
    root_dir = './test_ocrdataset'
    data = OCRData(root_dir)
    data.makeNeededDir()
    print('The arguments :')
    print(data.args)
    
    data.saveTopNCharacters2File(data.args['characters_file_path'], data.args['classes_number'], data.args['id_character_file_path'])
    data.args['image_number'] = 20
    data.synthesizeAllImages(data.args['image_number'])


class OCRData(object):
    def __init__(self, root_dir):
        self.args = {}
        self.args['root_dir'] = root_dir
        args = self.setArguments()

    def setArguments(self):
        self.args['characters_length_tuple'] = (3, 6)
        self.args['valiation_rate'] = 0.2
        self.args['test_rate'] = 0.2
        self.args['background_image_dir'] = './background'
        self.args['fonts_dir'] = './fonts'
        self.args['characters_file_path'] = './characters.txt'
        self.args['classes_number'] = 5
        self.args['id_character_file_path'] = os.path.basename(self.args['characters_file_path']).split('.')[0] + '_top_%d.txt'%self.args['classes_number']
        self.args['font_size_min'] = 32
        self.args['image_number'] = 10
        self.args['save_full_image'] = 0
        return self.args


    def makeNeededDir(self):
        utils.makeDirectory(self.args['root_dir'])
        self.makePartDirs('train')
        if self.args['valiation_rate'] > 0:
            self.makePartDirs('valiation')
        if self.args['test_rate'] > 0:
            self.makePartDirs('test')
        self.args['annotations_dir'] = os.path.join(self.args['root_dir'], 'annotations')
        utils.makeDirectory(self.args['annotations_dir'])
        return


    def makePartDirs(self, part_role):
        if self.args['save_full_image']:
            self.args[''.join([part_role,'_image_dir'])] = os.path.join(self.args['root_dir'], part_role+'_image')
            utils.makeDirectory(self.args[part_role+'_image_dir'])
        self.args[part_role+'_part_image_dir'] = os.path.join(self.args['root_dir'], part_role+'_part_image')
        utils.makeDirectory(self.args[part_role+'_part_image_dir'])
        return
        
        
    def synthesizeAllImages(self, image_number):
        self.bg_img_list = utils.getBackgroundListFromDir(self.args['background_image_dir'])
        self.font_list = utils.getFontListFromDir(self.args['fonts_dir'])
        for i in tqdm.tqdm(range(image_number)):
            content, content_index = utils.get_contents(self.id_character_dict, self.args['characters_length_tuple'])
            background_image_path, font_path = map(utils.getRandomOneFromList, [self.bg_img_list, self.font_list])
            image, points = self.putContent2Image(content, background_image_path, font_path)
            if self.args['save_full_image']:
                self.saveImage(image, i)
            part_images, roi_points = utils.cropImageByPoints(image, points)
            self.saveImage(part_images, i, is_part=1)
            self.saveAnnotation(content_index, points, i)
        return
    
    
    def saveImage(self, image, image_index, is_part=0):
        image_save_dir = self.chooseSaveDirByIndex(image_index, is_part)
        utils.saveImage2Dir(image, image_save_dir, image_name=str(image_index))

    
    def saveAnnotation(self, content_index, points, image_index):
        for index, one_content in enumerate(content_index):
            ann_name = ''.join([str(image_index), '_', str(index), '.txt'])
            ann_path = os.path.join(self.args['annotations_dir'], ann_name)
            rectangle_points = utils.getOneLineRectanglePoints(points[index])
            with codecs.open(ann_path, 'w', encoding='utf-8') as file:
                file.write(' '.join([ann_name.split('.')[0], str(rectangle_points.tolist()), str(one_content)]))
            
    def putContent2Image(self, mulcontents, background_image_path, font_path, add_rectangle=0):
        try:
            image = Image.open(background_image_path)
            mulcontents_points = []
            font_size_max = image.size[0]/self.args['characters_length_tuple'][1]
            while font_size_max < self.args['font_size_min']:
                image = image.resize((image.size[0]*2, image.size[1]*2))
                font_size_max = image.size[0]/self.args['characters_length_tuple'][1]
            font_size = random.randint(self.args['font_size_min'], font_size_max)
            left_center_point= (random.randint(0, image.size[0]-font_size*max([len(i) for i in mulcontents])), random.randint(font_size*len(mulcontents), image.size[1]-font_size*len(mulcontents)/2))
            color = utils.setColor(image)
            for content in mulcontents:
                content_points = []
                self.txt_center_line = 0
                for character in content:
                    image, points = self.putOneCharacter2Image(character, image, font_path, font_size, left_center_point, color)
                    content_points.append(points)
                    left_center_point = (max(points[1][0], points[2][0]), left_center_point[1])
                left_center_point = utils.getNewLeftCenterPointByContentPoints(content_points)
                mulcontents_points.append(content_points)
            if add_rectangle == 1:
                image = utils.drawMulContentsRectangle(image, mulcontents_points)
            image_out = image
        except AssertionError:
            # print('MadeError, retry.')
            image_out, mulcontents_points = self.putContent2Image(mulcontents, background_image_path, font_path)
        return image_out, mulcontents_points
        

    def putOneCharacter2Image(self, character, background_image, font_path, font_size, left_center_point, color=None):
        background = background_image.convert('RGBA')
        font = ImageFont.truetype(font_path, font_size)
        width, height = font.getsize(character)
        
        txt = Image.new('RGBA', (width, height), (255,255,255,0))
        points_in_txt = utils.getPointsOfImageRectangle(width, height)
        draw = ImageDraw.Draw(txt)
        draw.text((0, 0), character, font=font, fill=(255,255,255,255))  # draw text, full opacity
        
        txt, points_in_txt = utils.augmentImage(txt, points_in_txt)
        points = utils.mergeBgimgAndTxtimgPoints(left_center_point, points_in_txt)
        assert points[0][0] >= 0 and points[0][1] >= 0
        assert points[2][0] <= background.size[0] and points[2][1] <=background.size[1]
        # out_image = Image.alpha_composite(background, txt)
        out_image = utils.mergeImageAtPoint(background, txt, tuple(points[0]), color)
        out_image = out_image.convert('RGB')        
        return out_image, points
    
    
    def saveTopNCharacters2File(self, characters_file_path, top_n, save_path):
        if 'id_character_dict' not in dict():
            self.id_character_dict = utils.getTopNCharacters2Dict(characters_file_path, top_n)
        utils.saveIdCharacterDict2File(self.id_character_dict, save_path)
        return


    def chooseSaveDirByIndex(self, image_number_index, is_part_img=0):
        train_rate = 1 - self.args['test_rate'] - self.args['valiation_rate']
        if image_number_index < int(train_rate * self.args['image_number']):
            if not is_part_img:
                image_save_dir = self.args['train_image_dir']
            else:
                image_save_dir = self.args['train_part_image_dir']
        elif image_number_index < int((train_rate + self.args['valiation_rate'])* self.args['image_number']):
            if not is_part_img:
                image_save_dir = self.args['valiation_image_dir']
            else:
                image_save_dir = self.args['valiation_part_image_dir']
        else:
            if not is_part_img:
                image_save_dir = self.args['test_image_dir']
            else:
                image_save_dir = self.args['test_part_image_dir']
        return image_save_dir


if __name__ == '__main__':
    main()