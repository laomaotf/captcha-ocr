# coding=utf-8
############################################################################
#参考代码
#https://github.com/YunYang1994/yymnist/blob/master/make_data.py
############################################################################

import os
from PIL import Image,ImageDraw
import numpy as np
import shutil
from collections import defaultdict
import random

import json

with open("config.json","r",encoding='utf8') as f:
    config = json.load(f)

config_charset = config['charset']
config_captcha = config['captcha']

config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])
config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

#加载字符集文件名列表
charset_images = defaultdict(list)
for ind,c in enumerate(config_charset['chars']):
    char_root = os.path.join(config_charset['image_root'], "{}".format(c))
    images = [os.path.join(char_root, name) for name in os.listdir(char_root) if os.path.splitext(name)[-1] == '.jpg']
    charset_images[c] = images

if os.path.exists(config_captcha['image_root']): shutil.rmtree(config_captcha['image_root'])

os.makedirs(config_captcha['image_root'],exist_ok=True)

annotation_file_path = config_captcha['annotation_file']

#创建charset到label的映射
char2label = defaultdict(int)
for ind,c in enumerate(config_charset['chars']):
    char2label[c] = ind



def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""

    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    A3 = (xmax-xmin) * (ymax - ymin)
    return  A3 / (A1 + A2 - A3 + 0.000001)


def make_image(data, char_path, char, size):

    char_index = len(data['chars'])

    canvas = data['canvas']
    mask = data['mask']



    char_w, char_h = size,size

    char_image = Image.open(char_path)
    char_image = char_image.resize((char_w,char_h), Image.NEAREST)
    #paste时只贴白色点
    alpha_channel = Image.fromarray(np.asarray(char_image) > 128)

    canvas_w, canvas_h = canvas.size

    block_width = canvas_w // config_captcha['max_text_length']


    add_new = False
    try_num = 0
    while try_num < 100:
        try_num += 1
        #if char_index + 1 > config_captcha['max_text_length']:
        #    break
        #xmin = random.randint(data['dx'], canvas_w-char_w)
        valid_xmin = block_width * char_index
        valid_xmax = min([block_width * (char_index + 1), canvas_w - char_w])
        if valid_xmin >= valid_xmax:
            break
        xmin = random.randint(valid_xmin, valid_xmax )
        ymin = random.randint(0, canvas_h-char_h)
        xmax = xmin + char_w
        ymax = ymin + char_h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in data['bboxes']]
        if iou == [] or max(iou) < 0.1:
            data['bboxes'].append(box)
            data['chars'].append(char)
            data['dx'] = xmax + 1
            add_new = True
            break
    if not add_new:
        return data

    canvas.paste(char_image,(xmin,ymin),alpha_channel)

    char_bin_image = char_image.convert("1")
    char_bin_image = np.where(np.asarray(char_bin_image), 255 * np.ones_like(char_bin_image), np.zeros_like(char_bin_image))
    char_bin_image = Image.fromarray(char_bin_image)
    mask.paste(char_bin_image, (xmin,ymin),alpha_channel)


    data['canvas'] = canvas
    data['mask'] = mask
    return data


def make_captcha_canvas(size):
    width,height = size
    canvas = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    canvas[:,:,0],canvas[:,:,1],canvas[:,:,2] = [random.randint(0,255) for _ in range(3)]
    canvas = Image.fromarray(canvas)
    draw = ImageDraw.ImageDraw(canvas)
    for _ in range(100):
        xy = (random.randint(0,width), random.randint(0,height))
        c = [random.randint(0, 255) for _ in range(3)]
        draw.point(xy,fill=tuple(c))
    return canvas


annotation_stat = {"image":[],"chars":[]}
with open(annotation_file_path, "w") as annotation_fd:
    image_num = 0
    while image_num < config_captcha['num_image']:

        width = random.randint(config_captcha['width'][0],config_captcha['width'][1])
        height = random.randint(config_captcha['height'][0], config_captcha['height'][1])
        canvas = make_captcha_canvas((width,height))
        mask = Image.fromarray(np.zeros(shape=[height,width],dtype=np.uint8))
        #canvas = np.zeros(shape=[height, width, 3],dtype=np.uint8)
        #canvas = Image.fromarray(canvas)
        empty_image = True

        annotation = {"canvas":canvas, "bboxes":[], "chars":[], 'dx':0, 'mask':mask}

        min_size = min([width,height]) #字符别缩放到min_size，然后再缩放到不同大小目标
        max_text_length = config_captcha['max_text_length']
        # 小尺度目标
        sizes = [int(min_size * r) for r in [0.1,0.2,0.3,0.4]]
        N = random.randint(0, int(max_text_length * config_captcha['ratio_each_size']['small']))
        if N != 0: empty_image = False
        for _ in range(N):
            size = random.choice(sizes)
            c = random.choice(list(charset_images.keys()))
            idx = random.randint(0, len(charset_images[c])-1)
            annotation = make_image(annotation, charset_images[c][idx], c, size)

        # 中尺度目标
        sizes = [int(min_size * r) for r in [0.45, 0.55,0.65]]
        N = random.randint(max_text_length - len(annotation['chars']), max_text_length)
        if N != 0: empty_image = False
        for _ in range(N):
            size = random.choice(sizes)
            c = random.choice(list(charset_images.keys()))
            idx = random.randint(0, len(charset_images[c])-1)
            annotation = make_image(annotation, charset_images[c][idx], c, size)

        # big object
        sizes = [int(min_size * r) for r in [0.7, 0.8, 0.9,0.98]]
        #n = int(max_text_length * config_captcha['ratio_each_size']['big'])
        N = random.randint(max_text_length - len(annotation['chars']), max_text_length)
        if N != 0: empty_image = False
        for _ in range(N):
            size = random.choice(sizes)
            c = random.choice(list(charset_images.keys()))
            idx = random.randint(0, len(charset_images[c]) - 1)
            annotation = make_image(annotation, charset_images[c][idx], c, size)

        if empty_image: continue
        if len(annotation['chars']) < config_captcha['min_text_length'] or len(annotation['chars']) > config_captcha['max_text_length']:
            continue

        if 0:
            draw = ImageDraw.ImageDraw(annotation['canvas'])
            for bbox,c in zip(annotation['bboxes'], annotation['chars']):
                draw.rectangle(bbox,outline=(255,0,0),width=3)
            #annotation['canvas'].show()

        image_name = "%08d.jpg"%(image_num+1)
        image_path = os.path.join(config_captcha['image_root'],image_name)
        annotation['canvas'].save(image_path)

        mask_name = "%08d.mask.png"%(image_num+1)
        mask_path = os.path.join(config_captcha['image_root'], mask_name)
        annotation['mask'].save(mask_path)

        annotation_stat['image'].append(image_name)
        annotation_stat['chars'].append(len(annotation['chars']))

        line = ["{}".format(1 if image_num < config_captcha['num_train_image'] else 0)]
        line += [image_name]
        labels = list(map(lambda c: char2label[c], annotation['chars']))
        for k in range(len(labels)):
            bbox = annotation['bboxes'][k]
            one = '{} {} {} {} {}'.format(bbox[0],bbox[1],bbox[2],bbox[3],labels[k])
            line.append(one)

        image_num += 1
        annotation_fd.write(','.join(line) + "\n")


import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(annotation_stat)
print(df['chars'].describe())
df['chars_counts'] = df['chars'].value_counts()
df['chars_counts'].plot.pie()
plt.show()