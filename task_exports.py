# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from PIL import Image,ImageDraw
from networks.network import CNN,UNET
import json
import warnings
import random
import matplotlib.pyplot as plt
import cv2


#task toy的任务
class CLASS_TASK_TOY(CNN):
    def __init__(self, charsets, max_text_len, input_size, model_folder):
        super(CLASS_TASK_TOY, self).__init__(max_text_len * len(charsets))

        self.input_height, self.input_width = input_size
        self.max_text_len = max_text_len
        self.charsets = charsets
        latest_ckpt = tf.train.latest_checkpoint(model_folder)
        if latest_ckpt:
            super(CLASS_TASK_TOY,self).load_weights(latest_ckpt)
        else:
            warnings.warn_explicit("no model found!!!")
            return

    def run(self, imgs):

        if not isinstance(imgs, list):
            imgs = [imgs]

        datas = []

        for img in imgs:

            width,height = img.width, img.height

            if height != self.input_height or width != self.input_width:
                img = img.resize((self.input_width, self.input_height), Image.CUBIC)

            img_data = np.array(img,dtype=np.uint8)

            data = tf.image.convert_image_dtype(img_data, tf.float32)
            data = tf.reshape(data, (self.input_height,self.input_width, -1))
            datas += [data]

        data = tf.stack(datas)
        predict_output = super().predict(data)
        predict_output = np.reshape(predict_output, (len(datas),self.max_text_len, -1))
        predict_output = np.argmax(predict_output, axis=-1)
        predict_text = []
        for batch in range(len(datas)):
            text = []
            for field in range(self.max_text_len):
                char = predict_output[batch, field]
                text.append(self.charsets[char])
            text = ''.join(text)
            predict_text += [text.strip()] #刪除空格

        # 返回识别结果
        return predict_text

# task segment的任务
class CLASS_TASK_SEGMENT:
    def __init__(self, input_size, num_classes, model_folder):
        self.input_height, self.input_width = input_size
        self.num_classes = num_classes
        self.net = UNET(input_shape=(self.input_height, self.input_width,3),
                        num_classes = num_classes + 1)
        latest_ckpt = tf.train.latest_checkpoint(model_folder)
        if latest_ckpt:
            self.net.load_weights(latest_ckpt)
        else:
            warnings.warn_explicit("no model found!!!")
            return
    def convert_mask_to_bbox(self,masks):
        bboxes_all = []
        for mask in masks:
            if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8:
                mask = np.asarray(mask, np.uint8)
            objects, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = []
            for object in objects:
                x,y,w,h = cv2.boundingRect(object)
                # 删除小目标
                if w * h < 50:
                    continue

                #mnist是正方形目标，为了提高后续OCR准确率，需要扩展bbox
                cx, cy = x + w/2, y + h/2
                r = max([w,h])/2
                x0, x1 = int(cx - r), int(cx + r)
                y0, y1 = int(cy - r), int(cy + r)
                bboxes += [ (x0,y0,x1,y1)  ]
            bboxes_all.append( bboxes )
        return bboxes_all


    def run(self, imgs, return_bbox = True):

        if not isinstance(imgs, list):
            imgs = [imgs]

        datas = []
        org_hw = []
        for img in imgs:
            if not isinstance(img,np.ndarray):
                img = np.asarray(img,dtype=np.uint8)
            h, w = img.shape[0], img.shape[1]
            org_hw += [ (h,w) ]
            img = tf.image.resize_with_crop_or_pad(img, self.input_height, self.input_width)

            data = tf.image.convert_image_dtype(img, tf.float32)
            #data = tf.reshape(data, (self.input_height, self.input_width, -1))
            datas += [data]

        data = tf.stack(datas)
        predict_output = self.net.predict(data)
        predict_output = tf.argmax(predict_output, axis=-1)
        # 通过裁剪恢复原始尺寸
        masks = []
        predict_output = np.asarray(predict_output)
        for batch in range(predict_output.shape[0]):
            H, W = predict_output[batch].shape
            y0, x0 = (H - org_hw[batch][0]) // 2, (W - org_hw[batch][1]) // 2
            y1, x1 = y0 + org_hw[batch][0], x0 + org_hw[batch][1]
            masks.append( predict_output[0,y0:y1, x0:x1] ) #tf的组织结果是BHWC
        # 返回识别结果
        if return_bbox:
            return masks,self.convert_mask_to_bbox(masks)
        return masks


#task OCR的任务
class CLASS_TASK_OCR(CNN):
    def __init__(self, charsets, input_size, model_folder):
        super(CLASS_TASK_OCR, self).__init__(len(charsets))

        self.input_height, self.input_width = input_size
        self.charsets = charsets
        latest_ckpt = tf.train.latest_checkpoint(model_folder)
        if latest_ckpt:
            super(CLASS_TASK_OCR,self).load_weights(latest_ckpt)
        else:
            warnings.warn_explicit("no model found!!!")
            return

    def run(self, imgs): #输入一副图中多个字符

        if not isinstance(imgs, list):
            imgs = [imgs]

        datas = []
        for img in imgs:
            if isinstance(img,np.ndarray):
                img = Image.fromarray(img)
            width,height = img.width, img.height
            if height != self.input_height or width != self.input_width:
                img = img.resize((self.input_width, self.input_height), Image.CUBIC)
            img_data = np.array(img,dtype=np.uint8)
            data = tf.image.convert_image_dtype(img_data, tf.float32)
            #data = tf.reshape(data, (self.input_height,self.input_width, -1))
            datas.append(data)
        data = tf.stack(datas,axis=0)
        predict_output = super().predict(data)
        predict_output = np.argmax(predict_output, axis=-1)
        predict_text = []
        for ind in predict_output:
            predict_text.append(self.charsets[ind])
        predict_text = ''.join(predict_text)

        return predict_text
###########################################################################
#exports
#task_toy部署
class CLASS_TOY(object):
    def __init__(self,config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        project_root = config['project_root']

        config_task = config['task_toy']
        config_captcha = config['captcha']
        max_text_len = config_captcha['max_text_length']
        #image_root = os.path.join(project_root, config_captcha['image_root'])
        input_size = [config_task['input_height'], config_task['input_width']]
        model_folder = os.path.join(project_root, config_task['output_folder'], 'models')
        charset = config['charset']['chars']

        if config_captcha['max_text_length'] != config_captcha['min_text_length']:
            charset += [' ']

        self.net_toy = CLASS_TASK_TOY(charset, max_text_len, input_size, model_folder)


        return

    def run(self,images):
        text_all = self.net_toy.run(images)
        return ','.join(text_all) #逗號分割


class CLASS_SEGMENT_OCR(object):
    def __init__(self,config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        project_root = config['project_root']

        config_ocr = config['task_ocr']
        config_segment = config['task_segment']
        config_charset = config['charset']

        charset = config_charset['chars']
        #image_root = os.path.join(project_root, config_captcha['image_root'])

        input_size_ocr = [config_ocr['input_height'], config_ocr['input_width']]
        model_folder_ocr = os.path.join(project_root, config_ocr['output_folder'], 'models')

        input_size_segment = [config_segment['net_input_size'], config_segment['net_input_size']]
        model_folder_segment = os.path.join(project_root, config_segment['output_folder'], 'models')
        num_classes_segment = config_segment['num_classes']

        self.net_ocr = CLASS_TASK_OCR(charset, input_size_ocr, model_folder_ocr)
        self.net_segment = CLASS_TASK_SEGMENT(input_size_segment, num_classes_segment, model_folder_segment)
        return

    def run(self,images):
        if not isinstance(images,list):
            images = [images]

        masks, bboxes_all = self.net_segment.run(images)
        text_all = []
        for image,mask, bboxes in zip(images,masks, bboxes_all):

            #bboxes按照x从小到大排序，和captcha生成方法有关系
            bboxes = sorted(bboxes, key=lambda x:x[0])
            ##############################################
            images_char = []
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            for (x0,y0,x1,y1) in bboxes:
                images_char.append( image[y0:y1, x0:x1,:] )
            text = self.net_ocr.run(images_char)
            text_all.append(text)
        return ','.join(text_all)




###########################################################################
#TEST
def test_task_ocr():

    with open("../config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    project_root = config['project_root']

    config_task = config['task_ocr']
    config_captcha = config['captcha']

    image_root = os.path.join(project_root,config_captcha['image_root'])
    input_size = [config_task['input_height'], config_task['input_width']]
    model_folder = os.path.join(project_root,config_task['output_folder'],'models')
    charset = config['charset']['chars']

    R = CLASS_TASK_OCR(charset, input_size, model_folder)

    annotations = []
    annotation_file = os.path.join(project_root,config_captcha['annotation_file'])
    with open(annotation_file, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            usage = int(items[0])
            if usage == 1:
                continue  # 训练集数据
            path = os.path.join(image_root, items[1].strip())
            objects = list(map(lambda x: [int(y) for y in x.split(' ')], items[2:]))
            annotations.append((path, objects))

    figs,axes = plt.subplots(3,5)
    axes = axes.flatten()
    for k in range(len(axes)):
        image_path, objects = random.choice(annotations)
        image = Image.open(image_path)
        axes[k].imshow(np.asarray(image))

        images_char = []
        labels_gt = []
        for object in objects:
            x0, y0, x1, y1, label = [int(x) for x in object]
            labels_gt += [charset[label]]
            image_crop = image.crop((x0, y0, x1, y1))
            images_char.append( np.asarray(image_crop)  )

        predict_text = R.run(images_char)
        labels_gt = ''.join(labels_gt)
        if labels_gt != predict_text:
            axes[k].set_title("error: {} -> {}".format(labels_gt, predict_text))
            # print(gt_label,',',predict_text)
        else:
            axes[k].set_title("{}".format(labels_gt))
    plt.show()


def test_task_segment():
    with open("../config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    project_root = config['project_root']

    config_task = config['task_segment']
    config_captcha = config['captcha']
    pixel_scale = config_captcha['pixel2label']

    image_root = os.path.join(project_root, config_captcha['image_root'])
    input_size = [config_task['net_input_size'], config_task['net_input_size']]
    model_folder = os.path.join(project_root, config_task['output_folder'], 'models')
    num_classes = config_task['num_classes']
    R = CLASS_TASK_SEGMENT(input_size, num_classes, model_folder)

    annotations = []
    annotation_file = os.path.join(project_root, config_captcha['annotation_file'])
    with open(annotation_file, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            usage = int(items[0])
            if usage == 1:
                continue  # 训练集数据
            path = os.path.join(image_root, items[1].strip())
            objects = list(map(lambda x: [int(y) for y in x.split(' ')], items[2:]))
            annotations.append((path, objects))

    figs, axes = plt.subplots(3, 2)
    for k in range(axes.shape[0]):
        image_path, objects = random.choice(annotations)
        image = Image.open(image_path)

        masks,bboxes = R.run(image)
        draw = ImageDraw.ImageDraw(image)
        for x0,y0,x1,y1 in bboxes[0]:
            draw.rectangle((x0,y0,x1,y1),fill=None,outline="Red",width=3)
        axes[k, 0].imshow(np.asarray(image), cmap='gray')
        axes[k,1].imshow(np.asarray(masks[0]) * pixel_scale,cmap='gray')
    plt.show()



def test_segment_ocr():
    R = CLASS_SEGMENT_OCR('../config.json')

    with open("../config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    project_root = config['project_root']

    config_captcha = config['captcha']
    image_root = os.path.join(project_root, config_captcha['image_root'])

    annotations = []
    annotation_file = os.path.join(project_root, config_captcha['annotation_file'])
    with open(annotation_file, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            usage = int(items[0])
            if usage == 1:
                continue  # 训练集数据
            path = os.path.join(image_root, items[1].strip())
            objects = list(map(lambda x: [int(y) for y in x.split(' ')], items[2:]))
            annotations.append((path, objects))

    figs, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for k in range(len(axes)):
        image_path, objects = random.choice(annotations)
        image = Image.open(image_path)
        text = R.run(image)
        axes[k].imshow(np.asarray(image), cmap='gray')
        axes[k].set_title(text)
    plt.show()



if __name__ == '__main__':
    test_segment_ocr()
