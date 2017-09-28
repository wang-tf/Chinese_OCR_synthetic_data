#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import random
from PIL import Image, ImageDraw, ImageFont
import utils 
import codecs



def main():
    data = OCRData()
    args = data.setArguments()
    print(args)
    data.saveTopNCharacters2File(data.args['characters_file_path'], data.args['classes_number'], data.args['id_character_file_path'])
    data.bg_img_list = utils.getBackgroundListFromDir(data.args['background_image_dir'])
    print('Load backgroudn image: %d'%len(data.bg_img_list))
    data.font_list = utils.getFontListFromDir(data.args['fonts_dir'])
    data.args['image_number'] = 20
    data.synthesizeAllImages(data.args['image_number'])

    #data.producter(classes_num*800) # image muber for generating
    # print(content)


class OCRData(object):
    def __init__(self):
        self.setArguments()
        self.makeNeededDir()

    def setArguments(self):
        self.args = {}
        self.args['root_path'] = '/media/tengfei/Elements/synthetic_data'
        self.args['characters_length_tuple'] = (3, 6)
        self.args['valiation_rate'] = 0.2
        self.args['test_rate'] = 0.2
        self.args['background_image_dir'] = os.path.join(self.args['root_path'], 'background')
        self.args['fonts_dir'] = './fonts'
        self.args['characters_file_path'] = os.path.join(self.args['root_path'], 'characters.txt')
        self.args['classes_number'] = 5
        self.args['id_character_file_path'] = os.path.basename(self.args['characters_file_path']).split('.')[0] + '_top_%d.txt'%self.args['classes_number']
        self.args['font_size_min'] = 32
        self.args['image_number'] = 10
        return self.args


    def makeNeededDir(self):
        utils.makeDirectory(self.args['root_path'])
        self.train_image_dir = os.path.join(self.args['root_path'], 'train_img')
        self.valiation_image_dir = os.path.join(self.args['root_path'], 'valiation_image')
        self.test_image_dir = os.path.join(self.args['root_path'], 'test_image')
        utils.makeDirectory(self.train_image_dir)
        utils.makeDirectory(self.valiation_image_dir)
        utils.makeDirectory(self.test_image_dir)


    def synthesizeAllImages(self, image_number):
        for i in range(image_number):
            content, content_index = utils.get_contents(self.id_character_dict, self.args['characters_length_tuple'])
            background_image_path = self.bg_img_list[random.randint(0, len(self.bg_img_list)-1)]
            font_path = self.font_list[random.randint(0, len(self.font_list)-1)]
            image, points = self.putContent2Image(content, background_image_path, font_path)
            image_save_dir = self.chooseSaveDirByIndex(i)
            utils.saveImage2Dir(image, image_save_dir, image_name=str(i))
            print(points)
        return


    def putContent2Image(self, mulcontents, background_image_path, font_path):
        try:
            image = Image.open(background_image_path)
            mulcontents_points = []
            left_top_point= (random.randint(0, image.size[0]), random.randint(0, image.size[1]))
            font_size_max = image.size[0]/self.args['characters_length_tuple'][1]
            while font_size_max < self.args['font_size_min']:
                image = image.resize((image.size[0]*2, image.size[1]*2))
                font_size_max = image.size[0]/self.args['characters_length_tuple'][1]
            font_size = random.randint(self.args['font_size_min'], font_size_max)
            for content in mulcontents:
                content_points = []
                for character in content:
                    image, points = self.putOneCharacter2Image(character, image, font_path, font_size, left_top_point)  # points 为四个点的坐标
                    content_points.append(points)
                    left_top_point = points[1]
                left_top_point = content_points[0][3]
                mulcontents_points.append(content_points)
            image_out = image
        except AssertionError:
            # print('MadeError, retry.')
            image_out, mulcontents_points = self.putContent2Image(mulcontents, background_image_path, font_path)
        return image_out, mulcontents_points
        

    def putOneCharacter2Image(self, character, background_image, font_path, font_size, left_top_point):
        background = background_image.convert('RGBA')
        txt = Image.new('RGBA', background.size, (255,255,255,0))
        font = ImageFont.truetype(font_path, font_size)
        width, height = font.getsize(character)
        left, top = left_top_point
        points = [left_top_point, (left+width, top), (left+width, top+height), (left, top+height)]  # clockwise
        assert points[2][0] <= background.size[0] and points[2][1] <=background.size[1]
        
        draw = ImageDraw.Draw(txt)
        # draw text, full opacity
        draw.text(left_top_point, character, font=font, fill=(255,255,255,255))
        draw.rectangle([points[0],points[2]])
        print(txt)
        txt, points = utils.augmentImage(txt)
        print(txt)
        # out_image = Image.alpha_composite(background, txt)
        out_image = utils.mergeImageAtPint(background, txt, points[0])
        out_image = out_image.convert('RGB')
        return out_image, points


    def saveTopNCharacters2File(self, characters_file_path, top_n, save_path):
        file_path, file_name = os.path.split(characters_file_path)
        save_name = file_name.split('.')[0]+'_top_%d.txt'%top_n
        if 'id_character_dict' not in dict():
            self.id_character_dict = utils.getTopNCharacters2Dict(characters_file_path, top_n)
        with codecs.open(save_name, 'w', encoding='utf-8') as file:
            for (key, val) in self.id_character_dict.items():
                file.write(str(key) + ' ' + val + '\n')  # the first index must be zero
        return


    def chooseSaveDirByIndex(self, image_number_index):
        train_rate = 1 - self.args['test_rate'] - self.args['valiation_rate']
        if image_number_index < int(train_rate * self.args['image_number']):
            image_save_dir = self.train_image_dir
        elif image_number_index < int((train_rate + self.args['valiation_rate'])* self.args['image_number']):
            image_save_dir = self.valiation_image_dir
        else:
            image_save_dir = self.test_image_dir
        return image_save_dir


if __name__ == '__main__':
    main()