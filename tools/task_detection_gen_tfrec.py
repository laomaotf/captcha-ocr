# encoding=utf-8
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import json
import warnings

########################################################
#参考
#https://github.com/xingyizhou/CenterNet
##########################################################

plt.rcParams['font.sans-serif'] = ['SimHei']

#print(os.environ['CUDA_VISIBLE_DEVICES'])

INPUT_1, INPUT_2, INPUT_3, INPUT_4 = "data", "fm", "wh", "mask"

with open('../config.json', "r",encoding='utf8') as f:
    config = json.load(f)
config_captcha = config['captcha']
config_charset = config['charset']
config_task = config['task_detect']

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


def format_object(obj):
    x0,y0,x1,y1,label = [int(x.strip()) for x in obj.split(' ')]
    fm = [ (x0+x1)/2, (y0+y1)/2 ]
    wh = [x1-x0, y1-y0]
    return {"fm":fm, 'wh': wh, "class":0} #单类别

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, cls):
    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    #gaussian = np.expand_dims(gaussian,axis=-1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right,cls]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


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
    down_ratio = config_task['down_ratio']
    width, height = config_task['input_width'], config_task['input_height']
    for (path_img, objects) in tqdm(annotations):
        img = Image.open(path_img)
        img = np.array(img)
        H,W,C = img.shape
        if H != height or W != width:
            warnings.warn("目前不支持输入图裁剪和缩放，所以输入图尺寸必须满足 {}x{}".format(H,W))
            continue
        fm = np.zeros((H//down_ratio,W//down_ratio,config_task['num_classes']))
        wh = np.zeros((H//down_ratio,W//down_ratio,2))
        mask = np.zeros((H//down_ratio,W//down_ratio),dtype=np.int64)
        for obj in objects:
            info = format_object(obj)
            fx,fy = info['fm']
            ny,nx = int(fy)//down_ratio, int(fx)//down_ratio
            mask[ny,nx] = 1
            fm = draw_umich_gaussian(fm,(nx,ny),config_task['fm_radius'],info['class'])
            wh[ny,nx,0] = info['wh'][0]//down_ratio
            wh[ny, nx, 1] = info['wh'][1] // down_ratio

        # a--构造Features
        feature = dict()
        feature[INPUT_1] = _int64_feature(img.reshape([-1]))
        feature[INPUT_2] = _float_feature(fm.reshape([-1]))
        feature[INPUT_3] = _float_feature(wh.reshape([-1]))
        feature[INPUT_4] = _int64_feature(mask.reshape([-1]))
        features = tf.train.Features(feature=feature)
        # b--构造example
        example = tf.train.Example(features=features)
        # c--写入tfrecord
        writer.write(example.SerializeToString())  # write one SerializedString example each time

    #3--关闭tfrecord
    writer.close()
    print('%s is complete' % outfile)
    return outfile


def test_tfrd(tfrec_path):

    #把读取的example解析成的字典
    def _parse_function(example_proto):
        down_ratio = config_task['down_ratio']
        width, height = config_task['input_width'], config_task['input_height']
        features = {
            INPUT_1: tf.io.FixedLenFeature((height,width,3), tf.int64),
            INPUT_2: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio,config_task['num_classes']), tf.float32),
            INPUT_3: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio, 2),tf.float32),
            INPUT_4: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio, 1),tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)

        return parsed_features



    dataset = tf.data.TFRecordDataset(tfrec_path)

    total = 0
    for _ in dataset:
        total += 1
    print('{} : {}'.format(tfrec_path, total))

    dataset = dataset.map(_parse_function)
    for images_features in dataset.take(3):
        image_raw = images_features[INPUT_1].numpy().squeeze()
        fm = images_features[INPUT_2].numpy()[:,:,0].squeeze()

        figs,axes = plt.subplots(ncols=2,nrows=1)
        axes[0].imshow(fm,cmap="gray")
        axes[0].set_title("fm")
        axes[1].imshow(image_raw,cmap='gray')
        axes[1].set_title("image")
        plt.show()


def load_tfrd(tfrec_path):
    #把读取的example解析成的字典
    def _parse_function(example_proto):
        down_ratio = config_task['down_ratio']
        width, height = config_task['input_width'], config_task['input_height']
        features = {
            INPUT_1: tf.io.FixedLenFeature((height,width,3), tf.int64),
            INPUT_2: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio,config_task['num_classes']), tf.float32),
            INPUT_3: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio, 2),tf.float32),
            INPUT_4: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio, 1),tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)

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

