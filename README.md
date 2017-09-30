# Chinese_OCR_synthetic_data
---
## The progress was used to generate synthetic dataset for Chinese OCR.
Here we used [Augmenter](https://github.com/mdbloice/Augmentor) to augment out output characters in images, including rotate, skew, shear and distort.
And you can change characters.txt file to use other characters.
The main function can be found in the synthetic_data.py file.

The python package you may need:
- tqdm
- PIL(pillow)
- pathlib
- cv2(opencv)
- numpy
- codecs
- glob

---
## 本程序用于合成中文OCR数据库。
本程序使用了[Augmenter](https://github.com/mdbloice/Augmentor)库，以对输出的图像进行增强图片中的文本，其中包括旋转、倾斜、剪切和扭曲。这些形变的参数可以在utils.py中找到并修改。
在characters.txt中存放着所有的中文字符，如果想更换训练的字符请替换该文件。
main函数在synthetic_data.py中，可以按需要做修改。

使用之前可能需要安装一下的包：
- tqdm
- PIL(pillow)
- pathlib
- cv2(opencv)
- numpy
- codecs
- glob


![test](https://github.com/wang-tf/Chinese_OCR_synthetic_data/blob/master/test_ocrdataset/train_part_image/0_0.jpg)
