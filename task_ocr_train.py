# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os,shutil
from networks.network import CNN
from datetime import datetime


# tf.compat.v1.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )




class TrainModel:
    def __init__(self, config, config_charset, config_captcha, config_task):
        # 训练相关参数
        self.epoch_stop = config_task['epochs']

        self.train_batch_size = config_task['batch_size']
        self.test_batch_size = config_task['batch_size']

        self.train_tfrec_file = config_task['train_tfrec_file']
        self.test_tfrec_file = config_task['test_tfrec_file']

        self.image_root = config_captcha['image_root']
        self.annotation_file = config_captcha['annotation_file']

        # 获得图片宽高和字符长度基本信息
        self.image_height, self.image_width, self.image_channel = config_task['input_height'],config_task['input_width'],3

        self.output_folder = config_task['output_folder']
        self.output_model_folder = os.path.join(self.output_folder, "models")
        self.output_log_folder = os.path.join(self.output_folder, "logs",datetime.now().strftime("%Y%m%d-%H%M"))


        # 相关信息打印
        self.chars = config_charset['chars']
        self.char_set_len = len(config_charset['chars'])
        print("-->图片尺寸: {} X {}".format(self.image_height, self.image_width))
        print("-->验证码共{}类".format(self.char_set_len))
        print("-->使用测试集为 {}".format(config_task['train_tfrec_file']))
        print("-->使验证集为 {}".format(config_task['test_tfrec_file']))


    def load_train_test(self):
        # 把读取的example解析成的字典
        def _parse_function(example_proto):
            features = {'data': tf.io.FixedLenFeature((self.image_height, self.image_width, self.image_channel), tf.int64),
                        'label': tf.io.FixedLenFeature([1], tf.int64)}
            parsed_features = tf.io.parse_single_example(example_proto, features)
            parsed_features['data'] = tf.cast(parsed_features['data'], tf.uint8)
            return parsed_features

        dataset_train = tf.data.TFRecordDataset(self.train_tfrec_file)
        dataset_test = tf.data.TFRecordDataset(self.test_tfrec_file)

        dataset_train = dataset_train.map(_parse_function)
        dataset_test = dataset_test.map(_parse_function)

        #dataset_test.repeat()
        #dataset_train.repeat()
        return dataset_train, dataset_test


    def train_cnn(self):

        os.makedirs(self.output_model_folder,exist_ok=False)
        os.makedirs(self.output_log_folder, exist_ok=True)
        ##########################################
        #tensorboard的value单独放一个目录
        file_writer = tf.summary.create_file_writer( os.path.join(self.output_log_folder, "params") )
        file_writer.set_as_default()

        model = CNN(num_classes=self.char_set_len)

        def current_lr(epoch):
            base_lr = 1e-4
            lr = (1 - epoch * 1.0 / self.epoch_stop) * base_lr
            tf.summary.scalar('learning rate', data=lr, step=epoch)
            return lr

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
            metrics = ["categorical_accuracy","accuracy"]
        )

        train_dataset, test_dataset = self.load_train_test()

        # 数据转换和增广
        # 输入的是tfrecord中的example，输出是二元组
        def convert_val(input_data):
            image, label = input_data['data'], input_data['label']
            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
            #image_vis = tf.reshape(image,(1,self.image_height,self.image_width,3))
            #tf.summary.image("ValImage", image_vis, step=0)
            return (image, label)

        def convert(input_data):
            image, label = input_data['data'], input_data['label']
            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
            #image_vis = tf.reshape(image,(1,self.image_height,self.image_width,3))
            #tf.summary.image("TrainingImage-RAW", image_vis, step=0)
            return (image, label)

        def augment(input_data):
            image, label = convert(input_data)
            #image = tf.image.resize_with_crop_or_pad(image, self.image_height + 8, self.image_width + 8)  # Add pixels of padding
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

            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]


            #image_vis = tf.reshape(image,(1,self.image_height,self.image_width,3))
            #tf.summary.image("TrainingImage-AUG", image_vis, step=0)

            return (image, label)

        # dataset转换成batches
        # 这里调用函数把tfrecord转换成(data,label)二元组，作为后续fit的输入
        train_batches = (
            train_dataset
                .shuffle( self.train_batch_size * 10 )
                .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                #.map(convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size=self.train_batch_size,drop_remainder=True)
                #.repeat() #应对2.0.0的issues(StartAbort Out of range: End of sequence),fit()中需要设置steps_per_epoch
                .prefetch(tf.data.experimental.AUTOTUNE)
        )

        test_batches = (
            test_dataset
                .map(convert_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size=self.test_batch_size,drop_remainder=True)
               # .repeat()
        )

        # loading checkpoints if existing
        ckpt_path = os.path.join(self.output_model_folder,"cp-{epoch:04d}.ckpt")
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
            history = model.fit(train_batches, epochs=self.epoch_stop ,
                                validation_data=test_batches, #valid data没设设置steps_per_epoch!!!
                                #steps_per_epoch = len(self.train_images_list) // self.train_batch_size, #配合repeat使用
                                #steps_per_epoch=1,
                                verbose=False,callbacks=callbacks)
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=self.epoch_stop + 1))
            print("save model before quit!")
        model.save_weights(ckpt_path.format(epoch=self.epoch_stop + 1))
        #model.save(os.path.join(os.path.dirname(ckpt_path), 'final.h5'))
        # tf.saved_model.save(model, os.path.join(self.output_folder,'final'))




    def test_images(self):
        model = CNN(num_classes = self.char_set_len)
        ckpt_path = os.path.join(self.output_model_folder,"cp-{epoch:04d}.ckpt")
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
        if latest_ckpt:
            start_epoch = int(os.path.basename(latest_ckpt).split('-')[1].split('.')[0])
            model.load_weights(latest_ckpt)
            print('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
        else:
            return

        annotations = []
        with open(self.annotation_file,'r') as f:
            for line in f:
                items = line.strip().split(',')
                usage = int(items[0])
                if usage == 1:
                    continue #训练集数据
                path = os.path.join(self.image_root,items[1].strip())
                objects = list(map(lambda x: [int(y) for y in x.split(' ')], items[2:]))
                annotations.append((path, objects))
        print("images in dataset: {}".format(len(annotations)))
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
                labels_gt += [self.chars[label]]

                image_crop = image.crop((x0,y0,x1,y1))
                image_crop = image_crop.resize((self.image_width,self.image_height),Image.BILINEAR)
                image_crop = np.asarray(image_crop)

                data = tf.image.convert_image_dtype(image_crop, tf.float32)
                data = tf.reshape(data,(self.image_height,self.image_width,-1))
                images_char += [data]
            labels_gt = "".join(labels_gt)

            data = tf.stack(images_char,axis=0)

            predict_output = model.predict(data)
            predict_output = np.argmax(predict_output,axis=-1)
            predict_text = []
            for ind in predict_output:
                predict_text.append(self.chars[ind])
            predict_text=''.join(predict_text)
            predict_text = predict_text.strip()
            if labels_gt != predict_text:
                axes[k].set_title("error: {} -> {}".format(labels_gt, predict_text))
                #print(gt_label,',',predict_text)
            else:
                axes[k].set_title("{}".format(labels_gt))

        plt.show()


def main():
    with open("config.json", "r",encoding='utf8') as f:
        config = json.load(f)
    config_captcha = config['captcha']
    config_charset = config['charset']
    config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])
    config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

    config_task = config['task_ocr']
    config_task['train_tfrec_file'] = os.path.join(config['project_root'], config_task['train_tfrec_file'])
    config_task['test_tfrec_file'] = os.path.join(config['project_root'], config_task['test_tfrec_file'])
    config_task['output_folder'] = os.path.join(config['project_root'], config_task['output_folder'])


    tm = TrainModel(config, config_charset, config_captcha, config_task)
    #tm.train_cnn()  # 开始训练模型
    tm.test_images()  # 识别图片示例


if __name__ == '__main__':
    main()
