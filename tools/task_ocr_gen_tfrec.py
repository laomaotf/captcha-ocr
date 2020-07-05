
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import json
import warnings

import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']

#print(os.environ['CUDA_VISIBLE_DEVICES'])

INPUT_1, INPUT_2 = "data", "label"

with open('../config.json', "r",encoding='utf8') as f:
    config = json.load(f)
config_captcha = config['captcha']
config_charset = config['charset']
config_task = config['task_ocr']

config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])
config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

config_task['train_tfrec_file'] = os.path.join(config['project_root'], config_task['train_tfrec_file'])
config_task['test_tfrec_file'] = os.path.join(config['project_root'], config_task['test_tfrec_file'])

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def crop_one_object(image,object):
    x0,y0,x1,y1,label = [int(x) for x in object]
    crop_image = image[y0:y1,x0:x1,:]
    return crop_image, label


def convert_to_tfrecord(split):
    if split == 'train':
        image_usage_req = 1
        outfile = config_task['train_tfrec_file']
    else:
        image_usage_req = 0
        outfile = config_task['test_tfrec_file']
    if not os.path.exists(os.path.dirname(outfile)): os.makedirs(os.path.dirname(outfile),exist_ok=True)
    if os.path.exists(outfile): os.remove(outfile)
    print('making %s record...' % outfile)
    #1-创建tfrecord
    writer = tf.io.TFRecordWriter(outfile)
    #2-解析数据，转成tfrecord
    annotations = []
    with open(config_captcha['annotation_file'], 'r') as f:
        for line in f:
            items = line.split(',')
            image_usage = int(items[0])
            if image_usage != image_usage_req:
                continue
            image_path = os.path.join(config_captcha['image_root'], items[1].strip())
            annotations.append( (image_path, items[2:]) )

    labels_list = []
    width, height = config_task['input_width'], config_task['input_height']
    for (path_img, objects) in tqdm(annotations):
        img = Image.open( path_img )
        img = np.array( img )

        for obj in objects:
            crop_image, label = crop_one_object(img,obj.split(' '))
            labels_list.append(label)

            crop_image = Image.fromarray(crop_image)
            crop_image = crop_image.resize((width,height),resample=Image.BICUBIC)
            crop_image = np.reshape( np.asarray(crop_image), [-1])
            #a--构造Features
            feature = dict()
            feature[INPUT_1] = _int64_feature(crop_image)
            feature[INPUT_2] = _int64_feature([label])
            features = tf.train.Features(feature=feature)
            #b--构造example
            example = tf.train.Example(features=features)
            #c--写入tfrecord
            writer.write(example.SerializeToString()) #write one SerializedString example each time
    #3--关闭tfrecord
    writer.close()
    print('%s is complete' % outfile)


    df = pd.DataFrame({"labels":labels_list})
    print(df.describe())
    df['labels_count'] = df['labels'].value_counts()
    df['labels_count'].plot.pie()
    plt.show()
    return outfile


def test_tfrd(tfrec_path):
    #把读取的example解析成的字典
    def _parse_function(example_proto):
        features = {INPUT_1: tf.io.FixedLenFeature((config_task['input_height'],config_task['input_width'],3), tf.int64),
                    INPUT_2: tf.io.FixedLenFeature([1], tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        #图像数据转换成uint8(候选显示使用)


        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)

        #for i in parsed_features:
        #    parsed_features[i] = tf.cast(parsed_features[i], tf.uint8)
        return parsed_features



    dataset = tf.data.TFRecordDataset(tfrec_path)

    total = 0
    for _ in dataset:
        total += 1
    print('{} : {}'.format(tfrec_path, total))

    dataset = dataset.map(_parse_function)
    for images_features in dataset.take(3):
        image_raw = images_features[INPUT_1].numpy().squeeze()
        label = images_features[INPUT_2].numpy()

        figs,axes = plt.subplots(nrows=1,ncols=1)
        axes.imshow(image_raw,cmap='gray')
        axes.set_title("{}".format(label))
        plt.show()


def load_tfrd(tfrec_path):
    #把读取的example解析成的字典
    def _parse_function(example_proto):
        features = {INPUT_1: tf.io.FixedLenFeature((config_task['input_height'],config_task['input_width'],3), tf.int64),
                    INPUT_2: tf.io.FixedLenFeature([1], tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        #图像数据转换成uint8(候选显示使用)


        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)
        #parsed_features[INPUT_2] = tf.cast(parsed_features[INPUT_2], tf.uint8)

        #for i in parsed_features:
        #    parsed_features[i] = tf.cast(parsed_features[i], tf.uint8)
        return parsed_features



    dataset = tf.data.TFRecordDataset(tfrec_path)

    dataset = dataset.map(_parse_function)
    return dataset




if os.path.exists(config_task['train_tfrec_file']) and os.path.exists(config_task['test_tfrec_file']):
    print("reuse tfrec")
    test_tfrd(config_task['test_tfrec_file'])
    test_tfrd(config_task['train_tfrec_file'])
else:
    convert_to_tfrecord("test")
    convert_to_tfrecord("train")

