
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import json
import warnings
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']

#print(os.environ['CUDA_VISIBLE_DEVICES'])

INPUT_1, INPUT_2 = "data", "label"


with open('../config.json', "r",encoding='utf8') as f:
    config = json.load(f)

config_captcha = config['captcha']
config_charset = config['charset']
config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])

config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

config['task_toy']['train_tfrec_file'] = os.path.join(config['project_root'], config['task_toy']['train_tfrec_file'])

config['task_toy']['test_tfrec_file'] = os.path.join(config['project_root'], config['task_toy']['test_tfrec_file'])

if config_captcha['max_text_length'] != config_captcha['min_text_length']:
    config_charset['chars'] += [' ']

char2label = defaultdict(int)
for k,c in enumerate(config_charset['chars']):
    char2label[c] = k



def text_to_onehot(text):
    """
    转标签为oneHot编码
    :param text: str
    :return: numpy.array
    """
    text_len = len(text)
    if text_len > config_captcha['max_text_length']:
        raise ValueError('验证码最长{}个字符'.format(config_captcha['max_text_length']))

    vector = np.zeros(len(char2label.keys()) * config_captcha['max_text_length'],np.int64)

    for i, ch in enumerate(text):
        idx = i * len(char2label.keys()) + char2label[ch]
        vector[idx] = 1
    return vector

def onehot_to_text(onehot):
    text = ""
    for k in range(0, len(onehot), len(char2label.keys())):
        code = onehot[k:k+len(char2label.keys())]
        for i in range(len(code)):
            if code[i] < 0.5:
                continue
            text += config_charset['chars'][i]
    text = text.replace(' ','#')
    return text


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




def convert_to_tfrecord(split):
    if split == 'train':
        image_usage_req = 1
        outfile = config['task_toy']['train_tfrec_file']
    else:
        image_usage_req = 0
        outfile = config['task_toy']['test_tfrec_file']
    if not os.path.exists(os.path.dirname(outfile)): os.makedirs(os.path.dirname(outfile),exist_ok=True)
    if os.path.exists(outfile): os.remove(outfile)

    print('making %s record...' % outfile)
    #1-创建tfrecord
    writer = tf.io.TFRecordWriter(outfile)
    #2-解析数据，转成tfrecord
    annotations = []
    with open(config_captcha['annotation_file'],'r') as f:
        for line in f:
            items = line.split(',')
            image_usage = int(items[0])
            if image_usage != image_usage_req:
                continue
            path = os.path.join(config_captcha['image_root'], items[1].strip())
            objects = list(map(lambda x: [int(y) for y in x.split(' ')], items[2:]))
            annotations.append((path, objects))


    input_width, input_height = config['task_toy']['input_width'],config['task_toy']['input_height']
    bar = tqdm(annotations)
    for (path,objects) in bar:
        bar.set_description(path)
        img = Image.open(path)
        img = img.resize((input_width, input_height),Image.BILINEAR)
        img = np.array( img )
        img = np.reshape(img, [-1])

        text = ""
        for object in objects:
            text += config_charset['chars'][object[-1]]
        for k in range(len(objects), config_captcha['max_text_length']):
            text += ' ' #填充space
        label = text_to_onehot(text)

        #a--构造Features
        feature = dict()
        feature[INPUT_1] = _int64_feature(img)
        feature[INPUT_2] = _int64_feature(label)
        features = tf.train.Features(feature=feature)
        #b--构造example
        example = tf.train.Example(features=features)
        #c--写入tfrecord
        writer.write(example.SerializeToString()) #write one SerializedString example each time
    #3--关闭tfrecord
    writer.close()
    print('%s is complete' % outfile)
    return outfile


def test_tfrd(tfrec_file):
    #把读取的example解析成的字典
    def _parse_function(example_proto):
        features = {INPUT_1: tf.io.FixedLenFeature((config['task_toy']['input_height'],config['task_toy']['input_width'],3), tf.int64),
                    INPUT_2: tf.io.FixedLenFeature([len(config_charset['chars']) * config_captcha['max_text_length']], tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        #图像数据转换成uint8(候选显示使用)

        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)
        #for i in parsed_features:
        #    parsed_features[i] = tf.cast(parsed_features[i], tf.uint8)
        return parsed_features



    dataset = tf.data.TFRecordDataset(tfrec_file)

    total = 0
    for _ in dataset:
        total += 1
    print('{} : {}'.format(tfrec_file, total))

    dataset = dataset.map(_parse_function)
    for images_features in dataset.take(3):
        image_raw = images_features[INPUT_1].numpy().squeeze()
        label = images_features[INPUT_2].numpy()
        label = onehot_to_text(label)

        plt.title(str(label))
        plt.imshow(image_raw,cmap='gray')
        plt.show()


def load_tfrd(tfrec_file):
    #把读取的example解析成的字典
    def _parse_function(example_proto):
        features = {INPUT_1: tf.io.FixedLenFeature((config['task_toy']['input_height'],config['task_toy']['input_width'],3), tf.int64),
                    INPUT_2: tf.io.FixedLenFeature([len(config_charset['chars']) * config_captcha['max_text_length']], tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        #图像数据转换成uint8(候选显示使用)

        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)
        #for i in parsed_features:
        #    parsed_features[i] = tf.cast(parsed_features[i], tf.uint8)
        return parsed_features

    dataset = tf.data.TFRecordDataset(tfrec_file)

    dataset = dataset.map(_parse_function)
    return dataset




if os.path.exists(config['task_toy']['train_tfrec_file']) and os.path.exists(config['task_toy']['test_tfrec_file']):
    print("reuse train.tfrecord and val.tfrecord")
    test_tfrd(config['task_toy']['test_tfrec_file'])
    test_tfrd(config['task_toy']['train_tfrec_file'])

else:
    convert_to_tfrecord('val')
    convert_to_tfrecord('train')
