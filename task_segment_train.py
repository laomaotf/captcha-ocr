# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os,shutil
from networks.network import UNET
from datetime import datetime


# tf.compat.v1.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )






def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


class TrainModel:
    def __init__(self, config, config_captcha, config_task, verify=False):
        # 训练相关参数
        self.epochs = config_task['epochs']

        self.annotation_file = config_captcha['annotation_file']
        self.image_root = config_captcha['image_root']

        # 获得图片宽高和字符长度基本信息
        self.image_height, self.image_width, self.image_channel = config_task['input_height'],config_task['input_width'],3

        self.net_input_size = config_task['net_input_size']

        self.train_tfrec_file, self.test_tfrec_file = config_task['train_tfrec_file'],config_task['test_tfrec_file']

        #self.config_task = config_task



        self.output_folder = config_task['output_folder']
        self.output_model_folder = os.path.join(self.output_folder, "models")
        self.output_log_folder = os.path.join(self.output_folder, "logs",datetime.now().strftime("%Y%m%d-%H%M"))



        self.num_train_image = config_captcha['num_train_image']
        self.num_test_image = config_captcha['num_image'] -config_captcha['num_train_image']

        self.num_classes = config_task['num_classes']

        self.train_batch_size = self.test_batch_size = config_task['batch_size']


        self.label_multipler = config_captcha['pixel2label']


        self.class_weight = {}
        for c in range(self.num_classes+1):
            self.class_weight[c] = config_task['class_weight']['{}'.format(c)]

    def load_trainval(self):

        def _parse_function(example_proto):
            features = {'data': tf.io.FixedLenFeature((self.image_height, self.image_width, 3), tf.int64),
                        'label': tf.io.FixedLenFeature([self.image_height,self.image_width,1], tf.int64)}
            parsed_features = tf.io.parse_single_example(example_proto, features)
            parsed_features['data'] = tf.cast(parsed_features['data'], tf.uint8)
            parsed_features['label'] = tf.cast(parsed_features['label'], tf.uint8)
            return parsed_features


        dataset_train = tf.data.TFRecordDataset(self.train_tfrec_file)
        dataset_val = tf.data.TFRecordDataset( self.test_tfrec_file)

        dataset_train = dataset_train.map(_parse_function)
        dataset_val = dataset_val.map(_parse_function)

        return dataset_train, dataset_val


    def train(self):
        os.makedirs(self.output_model_folder, exist_ok=False)
        os.makedirs(self.output_log_folder, exist_ok=True)
        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(self.image_height, self.image_width))
        print("-->类别数: {}".format(self.num_classes))
        print("-->类别权重:")
        for c in range(self.num_classes):
            print("\t{}:{:.6f}".format(c,self.class_weight[c]))

        file_writer = tf.summary.create_file_writer( os.path.join(self.output_log_folder, "params")) #子目录
        file_writer.set_as_default()

        model = UNET(input_shape=(self.net_input_size, self.net_input_size, self.image_channel),
                     num_classes=self.num_classes + 1) #add background

        def current_lr(epoch):
            base_lr = 1e-4
            lr = (1 - epoch * 1.0 / self.epoch_stop) * base_lr
            tf.summary.scalar('learning rate', data=lr, step=epoch)
            return lr

        #tf.keras.losses.sparse_categorical_crossentropy
        #tf.keras.metrics.categorical_accuracy
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss = ['sparse_categorical_crossentropy'],
            metrics = ["accuracy"] #通过字符串，直接调用函数!!!
        )

        train_dataset, val_dataset = self.load_trainval()

        # 数据转换和增广
        # 输入的是tfrecord中的example，输出是二元组
        def convert_val(input_data):
            image, label = input_data['data'], input_data['label']
            image = tf.image.resize_with_crop_or_pad(image, self.net_input_size,
                                                     self.net_input_size)  # Add pixels of padding
            label = tf.image.resize_with_crop_or_pad(label, self.net_input_size,
                                                     self.net_input_size)  # Add pixels of padding
            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
            #image_vis = tf.reshape(image,(1,self.image_height,self.image_width,3))
            #tf.summary.image("ValImage", image_vis, step=0)
            return (image, label)


        def augment(input_data):
            image, label = input_data['data'], input_data['label']
            #image = tf.image.random_crop(image, size=[self.image_height, self.image_width, 3])  # Random crop back to 60x100
            image = tf.image.random_contrast(image,lower=0.8, upper=1.3)  # Random brightness
            # image_vis = tf.reshape(image, (1, self.image_height, self.image_width, 3))
            # tf.summary.image("TrainingImage-AUG-1", image_vis, step=0)

            image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
            # image_vis = tf.reshape(image, (1, self.image_height, self.image_width, 3))
            # tf.summary.image("TrainingImage-AUG-2", image_vis, step=0)

            image = tf.image.random_saturation(image,lower=0.8,upper=1.3)
            # image_vis = tf.reshape(image, (1, self.image_height, self.image_width, 3))
            # tf.summary.image("TrainingImage-AUG-3", image_vis, step=0)

            image = tf.image.random_hue(image, max_delta=0.5)  # Random hue
            # image_vis = tf.reshape(image, (1, self.image_height, self.image_width, 3))
            # tf.summary.image("TrainingImage-AUG-4", image_vis, step=0)

            #image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_with_crop_or_pad(image, self.net_input_size,
                                                     self.net_input_size)  # Add pixels of padding
            label = tf.image.resize_with_crop_or_pad(label, self.net_input_size,
                                                     self.net_input_size)  # Add pixels of padding
            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]


            #image_vis = tf.reshape(image,(1,self.net_input_size,self.net_input_size,3))
            #tf.summary.image("train_image", image_vis, step=1)

            #label_vis = tf.reshape(label,(1,self.net_input_size,self.net_input_size,1)) * 100
            #tf.summary.image("train_label", label_vis, step=1)

            return (image, label)

        # dataset转换成batches
        # 这里调用函数把tfrecord转换成(data,label)二元组，作为后续fit的输入
        train_batches = (
            train_dataset
                #.shuffle(self.train_batch_size)
                .shuffle( self.num_train_image )
                .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size=self.train_batch_size,drop_remainder=True)
                #.repeat() #应对2.0.0的issues(StartAbort Out of range: End of sequence),fit()中需要设置steps_per_epoch
                .prefetch(tf.data.experimental.AUTOTUNE)
        )

        val_batches = (
            val_dataset
                .map(convert_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size=self.test_batch_size,drop_remainder=True)
        )

        # loading checkpoints if existing
        ckpt_path = os.path.join(self.output_model_folder,  "cp-{epoch:04d}.ckpt")  # 可以使用str.format
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
        if latest_ckpt:
            start_epoch = int(os.path.basename(latest_ckpt).split('-')[1].split('.')[0])
            model.load_weights(latest_ckpt)
            print('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
        else:
            print('passing resume since weights not there. training from scratch')
        # callback的一些配置可能影响训练速度
        # profile_batch of TensorBoard() will make training in eager model(slowing training), set zero to disable it
        callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, save_freq="epoch"),
                     tf.keras.callbacks.TensorBoard(self.output_log_folder, update_freq="epoch"),
                     #tf.keras.callbacks.LearningRateScheduler(current_lr),
                     ]
        try:
            #print(len( self.train_images_list) // self.train_batch_size)
            model.fit(train_batches, epochs=self.epochs ,
                                validation_data=val_batches, #valid data没设设置steps_per_epoch!!!
                                #steps_per_epoch = len(self.train_images_list) // self.train_batch_size, #配合repeat使用
                                #steps_per_epoch=1,
                                #设置类别权重
                                 class_weight = self.class_weight,
                                verbose=False,callbacks=callbacks)
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=self.epochs + 1))
            print("save model before quit!")
        model.save_weights(ckpt_path.format(epoch=self.epochs + 1))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'segment_final.h5'))

    def test_images(self):
        model = UNET(input_shape=(self.net_input_size, self.net_input_size, self.image_channel),
                     num_classes=self.num_classes + 1)
        ckpt_path = os.path.join(self.output_model_folder,  "cp-{epoch:04d}.ckpt")  # 可以使用str.format
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
        if latest_ckpt:
            start_epoch = int(os.path.basename(latest_ckpt).split('-')[1].split('.')[0])
            model.load_weights(latest_ckpt)
            print('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
        else:
            print('ERR: passing resume since weights not there')
            return 0

        annotations = []
        with open(self.annotation_file, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                usage = int(items[0])
                if usage == 1:
                    continue  # 训练集数据
                # 从图片名映射到对应的mask图片名
                mask_name = os.path.splitext(items[1].strip())[0] + ".mask.png"
                mask_path = os.path.join(self.image_root, mask_name)

                image_path = os.path.join(self.image_root, items[1].strip())
                annotations.append((image_path, mask_path))
        print("images in dataset: {}".format(len(annotations)))


        for k in range(3):
            path_img, path_mask = random.choice(annotations)
            captcha_mask = Image.open(path_mask)
            captcha_mask = np.array(captcha_mask, dtype=np.uint8)  # 向量化
            captcha_mask = np.expand_dims(captcha_mask,axis=-1)
            org_h, org_w, _ = captcha_mask.shape

            captcha_image = Image.open(path_img)
            captcha_array = np.array(captcha_image,dtype=np.uint8)  # 向量化



            captcha_array = tf.image.resize_with_crop_or_pad(captcha_array, self.net_input_size,self.net_input_size)


            data = tf.image.convert_image_dtype(captcha_array, tf.float32)
            data = tf.reshape(data,(1,self.net_input_size,self.net_input_size,-1))

            predict_output = model.predict(data)
            predict_output = tf.argmax(predict_output,axis=-1)
            predict_output = predict_output[...,tf.newaxis][0] * self.label_multipler

            #通过裁剪恢复原始尺寸
            predict_output = np.asarray(predict_output)
            H,W,C = predict_output.shape
            y0,x0 = (H-org_h)//2, (W-org_w)//2
            y1,x1 = y0 + org_h, x0 + org_w
            predict_output = predict_output[y0:y1, x0:x1,:]
            display([captcha_image,captcha_mask,predict_output])

        #plt.show()


def main():
    #https://tensorflow.google.cn/tutorials/images/segmentation?hl=zh_cn
    with open("config.json", "r", encoding='utf8') as f:
        config = json.load(f)
    config_captcha = config['captcha']
    #config_charset = config['charset']

    config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])
    config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

    config_task = config['task_segment']
    config_task['train_tfrec_file'] = os.path.join(config['project_root'], config_task['train_tfrec_file'])
    config_task['test_tfrec_file'] = os.path.join(config['project_root'], config_task['test_tfrec_file'])
    config_task['output_folder'] = os.path.join(config['project_root'], config_task['output_folder'])

    tm = TrainModel(config,config_captcha, config_task, verify=False)
    tm.train()  # 开始训练模型
    #tm.test_images()  # 识别图片示例


if __name__ == '__main__':
    main()
