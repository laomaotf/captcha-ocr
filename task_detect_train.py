# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image,ImageDraw
import random
import os
from networks.detection import get_detnet
from datetime import datetime

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


# tf.compat.v1.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )


######################################################################################
def focal_loss(alpha=0.25, gamma=2.0, eps=1e-6, from_logist=True):
    def _focal_loss(y_true, y_pred):
        if from_logist:
            pr_pred = math_ops.sigmoid(y_pred)
        else:
            pr_pred = y_pred

        #a.只有truth中心一个点作为gt
        pt = tf.where(y_true == 1, pr_pred, 1 - pr_pred) # 预测的置信度

        log_pt = tf.math.log(pt + eps) #转换成log，交叉熵定义
        #log_pt = math_ops.log(pt + eps)

        #b.相对focal loss额外增加的一个负样本权重
        neg_extra_weight = tf.pow(1-y_true,4) #对远离目标的点做额外惩罚
        #neg_extra_weight = math_ops.pow(1 - y_true, 4)

        weight = tf.where(y_true == 1, (1-alpha) * tf.ones_like(log_pt), alpha * tf.ones_like(log_pt) * neg_extra_weight)
        loss = -weight * tf.math.pow(1-pt, gamma)*log_pt
        #loss = -weight * math_ops.pow(1 - pt, gamma) * log_pt
        #return tf.reduce_mean(loss)
        return tf.reduce_sum(loss) / (tf.reduce_sum( tf.cast(y_true==1, tf.float32) ) + eps)
    return _focal_loss

def l1_loss(eps = 1e-6):
    def _l1_loss(y_true, y_pred, mask):
        loss = tf.abs(y_true - y_pred)
        mask2 = tf.concat([mask,mask],-1)
        loss = tf.where(mask2 == 1, loss, tf.zeros_like(loss))
        return tf.reduce_sum(loss) / (tf.reduce_sum(tf.cast(mask,tf.float32)) + eps)
    return _l1_loss

def centernet_loss():
    calc_fm_loss = focal_loss(from_logist=True)
    calc_wh_loss = l1_loss()
    def _loss(fm_true, fm_pred, wh_true, wh_pred, mask):
        fm_loss = calc_fm_loss(fm_true, fm_pred)
        wh_loss = calc_wh_loss(wh_true, wh_pred, mask)
        return fm_loss , wh_loss
    return _loss


class NMS:
    def __init__(self):
        self.net = tf.keras.Sequential()
        self.net.add(
            tf.keras.layers.MaxPool2D(pool_size=3,strides=1,padding='same')
        )
        return
    def __call__(self,data):
        out_data = self.net.predict(data)
        return tf.equal(data, out_data)


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

        self.train_tfrec_file, self.test_tfrec_file = config_task['train_tfrec_file'],config_task['test_tfrec_file']

        self.down_ratio = config_task['down_ratio']


        self.output_folder = config_task['output_folder']
        self.output_model_folder = os.path.join(self.output_folder, "models")
        self.output_log_folder = os.path.join(self.output_folder, "logs",datetime.now().strftime("%Y%m%d-%H%M"))

        self.finetune_flag = config_task['finetune_flag']


        self.num_train_image = config_captcha['num_train_image']
        self.num_test_image = config_captcha['num_image'] - config_captcha['num_train_image']

        self.num_classes = config_task['num_classes']

        self.train_batch_size = self.test_batch_size = config_task['batch_size']


    def load_trainval(self):

        def _parse_function(example_proto):
            features = {
                "data": tf.io.FixedLenFeature((self.image_height, self.image_width, 3), tf.int64),
                "fm": tf.io.FixedLenFeature((self.image_height // self.down_ratio, self.image_width // self.down_ratio, self.num_classes),
                                               tf.float32),
                "wh": tf.io.FixedLenFeature((self.image_height // self.down_ratio, self.image_width // self.down_ratio, 2), tf.float32),
                "mask": tf.io.FixedLenFeature((self.image_height // self.down_ratio, self.image_width // self.down_ratio, 1), tf.int64)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            parsed_features['data'] = tf.cast(parsed_features['data'], tf.uint8)
            return parsed_features


        dataset_train = tf.data.TFRecordDataset(self.train_tfrec_file)
        dataset_val = tf.data.TFRecordDataset( self.test_tfrec_file)

        dataset_train = dataset_train.map(_parse_function)
        dataset_val = dataset_val.map(_parse_function)

        return dataset_train, dataset_val


    def train(self):
        if not self.finetune_flag:
            os.makedirs(self.output_model_folder, exist_ok=False)
        os.makedirs(self.output_log_folder, exist_ok=True)

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(self.image_height, self.image_width))
        print("-->类别数: {}".format(self.num_classes))


        file_writer = tf.summary.create_file_writer( os.path.join(self.output_log_folder, "params")) #子目录
        file_writer.set_as_default()

        model = get_detnet(input_shape=(self.image_height, self.image_width, self.image_channel),
                           num_class=self.num_classes)

        tf.keras.utils.plot_model(
            model, to_file='mode_detect.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(1e-4),
        #     loss = ['sparse_categorical_crossentropy'],
        #     metrics = ["accuracy"] #通过字符串，直接调用函数!!!
        # )

        train_dataset, val_dataset = self.load_trainval()

        # 数据转换和增广
        # 输入的是tfrecord中的example，输出是二元组
        def convert_val(input_data):
            image, fm, wh,mask = input_data['data'], input_data['fm'], input_data['wh'], input_data['mask']
            image = tf.image.resize_with_crop_or_pad(image, self.image_height,
                                                     self.image_width)  # Add pixels of padding
            # fm = tf.image.resize_with_crop_or_pad(fm, self.image_height,
            #                                          self.image_width)  # Add pixels of padding
            #
            # wh = tf.image.resize_with_crop_or_pad(wh, self.image_height,
            #                                       self.image_width)  # Add pixels of padding
            #
            # mask = tf.image.resize_with_crop_or_pad(mask, self.image_height,
            #                                       self.image_width)  # Add pixels of padding

            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]

            return image, fm, wh,mask


        def augment(input_data):
            image, fm, wh,mask = input_data['data'], input_data['fm'], input_data['wh'], input_data['mask']

            image = tf.image.random_contrast(image,lower=0.8, upper=1.3)  # Random brightness

            image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness

            image = tf.image.random_saturation(image,lower=0.8,upper=1.3)

            image = tf.image.random_hue(image, max_delta=0.5)  # Random hue


            # image = tf.image.resize_with_crop_or_pad(image, self.image_height,
            #                                          self.image_width)  # Add pixels of padding
            # fm = tf.image.resize_with_crop_or_pad(fm, self.image_height,
            #                                       self.image_width)  # Add pixels of padding
            #
            # wh = tf.image.resize_with_crop_or_pad(wh, self.image_height,
            #                                       self.image_width)  # Add pixels of padding
            #
            # mask = tf.image.resize_with_crop_or_pad(mask, self.image_height,
            #                                         self.image_width)  # Add pixels of padding

            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image t

            return image, fm, wh,mask

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
            print('微调模型 {}, start epoch from {}'.format(latest_ckpt, start_epoch))
        else:
            print('随机初始化模型，start epoch from 0')

        calc_model_loss = centernet_loss()
        #################################
        #定义优化方案
        optimizer = tf.keras.optimizers.Adam(1e-3)
        try:
            for current_epoch in range(self.epochs):
                loss_in_epoch = 0
                loss_fm_in_epoch, loss_wh_in_epoch = 0, 0
                for iter_in_epoch,batch_train in enumerate(train_batches):
                    data, fm, wh, mask = batch_train
                    with tf.GradientTape() as tape:
                        #########################
                        #模型预测
                        pred_fm, pred_wh = model(data)
                        #########################
                        #计算loss
                        loss_fm, loss_wh = calc_model_loss(fm, pred_fm, wh, pred_wh, mask)
                        loss = loss_fm + loss_wh
                        loss_in_epoch += loss
                        loss_fm_in_epoch += loss_fm
                        loss_wh_in_epoch += loss_wh
                    ############################
                    #计算梯度
                    gradients = tape.gradient(loss, model.trainable_variables)
                    #############################
                    #更新模型
                    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                train_loss = loss_in_epoch/(self.train_batch_size * (iter_in_epoch + 1))
                train_fm_loss = loss_fm_in_epoch / (self.train_batch_size * (iter_in_epoch + 1))
                train_wh_loss = loss_wh_in_epoch / (self.train_batch_size * (iter_in_epoch + 1))

                loss_in_epoch = 0
                loss_fm_in_epoch, loss_wh_in_epoch = 0, 0
                for iter_in_epoch,batch_train in enumerate(val_batches):
                    data, fm, wh, mask = batch_train
                    pred_fm, pred_wh = model(data)

                    if iter_in_epoch < 10:
                        B,H,W,C = pred_fm.shape
                        one_pred = pred_fm[0].numpy()
                        m0,m1 = one_pred.min(), one_pred.max()
                        vis_pred = []
                        for c in range(C):
                            vis_pred.append(  (one_pred[:,:,c] - m0) * 255 / (m1 - m0 + 0.0001)   )
                        vis_pred = np.hstack(vis_pred).astype(np.uint8)
                        #vis_pred = Image.fromarray(vis_pred)

                        one_gt = fm[0].numpy()
                        m0, m1 = one_gt.min(), one_gt.max()
                        vis_gt = []
                        for c in range(C):
                            vis_gt.append((one_gt[:, :, 0] - m0) * 255 / (m1 - m0 + 0.0001))
                        vis_gt = np.hstack(vis_gt).astype(np.uint8)
                        #vis_gt = Image.fromarray(vis_gt)
                        vis = np.vstack((vis_gt, vis_pred))
                        vis = Image.fromarray(vis)
                        vis.save(os.path.join(self.output_folder,"debug_{}.jpg".format(iter_in_epoch)))


                    loss_fm, loss_wh = calc_model_loss(fm, pred_fm, wh, pred_wh, mask)
                    loss = loss_fm + loss_wh
                    loss_in_epoch += loss
                    loss_fm_in_epoch += loss_fm
                    loss_wh_in_epoch += loss_wh
                val_loss = loss_in_epoch/(self.test_batch_size * (iter_in_epoch + 1))
                val_fm_loss = loss_fm_in_epoch/(self.test_batch_size * (iter_in_epoch + 1))
                val_wh_loss = loss_wh_in_epoch / (self.test_batch_size * (iter_in_epoch + 1))

                tf.summary.scalar('TrainLoss', data=train_loss, step=current_epoch)
                tf.summary.scalar('TestLoss', data=val_loss, step=current_epoch)
                model.save_weights(ckpt_path.format(epoch=current_epoch))
                print("epoch {} train-loss {:.3e}({:.2e},{:.2e}) test-loss {:.3e}({:.2e},{:.2e})".format(
                    current_epoch + 1, train_loss,train_fm_loss,train_wh_loss,
                    val_loss,val_fm_loss, val_wh_loss))

        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=self.epochs + 1))
            print("save model before quit!")


        model.save_weights(ckpt_path.format(epoch=self.epochs + 1))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'segment_final.h5'))



    def test_images(self):
        model = get_detnet(input_shape=(self.image_height, self.image_width, self.image_channel),
                           num_class=self.num_classes)
        get_local_max = NMS()
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
                image_path = os.path.join(self.image_root, items[1].strip())
                annotations.append((image_path, items[2:]))
        print("images in dataset: {}".format(len(annotations)))


        for k in range(3):
            path_img, objs = random.choice(annotations)
            img = Image.open(path_img)
            data = np.array(img,dtype=np.uint8)

            data = tf.image.convert_image_dtype(data, tf.float32)
            data = tf.reshape(data,(1,self.image_height,self.image_width,-1))

            hm_pred, wh_pred = model.predict(data)
            hm_pred = tf.sigmoid(hm_pred)
         #   vis = hm_pred.numpy()[0,:,:,0] * 255
        #    plt.imshow(vis.astype(np.uint8),cmap='gray#')
        #    plt.show()
            mask_max = get_local_max(hm_pred)

            bboxes = []
            for cls in range(hm_pred.shape[-1]):
                for y in range(hm_pred.shape[1]):
                    for x in range(hm_pred.shape[2]):
                        prob = hm_pred[0,y,x,cls].numpy()
                        if prob < 0.1:
                            continue
                        if mask_max[0,y,x,cls] == False:
                            continue
                        w,h = wh_pred[0,y,x,0], wh_pred[0,y,x,1]
                        bboxes.append(  (x-w//2, y-h//2, x+w//2, y+h//2, prob)  )

            if len(bboxes) > 10:
                bboxes = sorted(bboxes,key = lambda x: x[-1], reverse=True)[0:10]

            draw = ImageDraw.ImageDraw(img)
            for (x0,y0,x1,y1,_) in bboxes:
                x0,x1 = int(x0 * self.down_ratio),int(x1 * self.down_ratio)
                y0,y1 = int(y0 * self.down_ratio), int(y1 * self.down_ratio)

                draw.rectangle((x0,y0,x1,y1),outline="Red",width=3)
            plt.imshow(np.asarray(img,dtype=np.uint8))
            plt.show()



def main():
    with open("config.json", "r", encoding='utf8') as f:
        config = json.load(f)
    config_captcha = config['captcha']

    config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])
    config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

    config_task = config['task_detect']
    config_task['train_tfrec_file'] = os.path.join(config['project_root'], config_task['train_tfrec_file'])
    config_task['test_tfrec_file'] = os.path.join(config['project_root'], config_task['test_tfrec_file'])
    config_task['output_folder'] = os.path.join(config['project_root'], config_task['output_folder'])

    tm = TrainModel(config,config_captcha, config_task, verify=False)
    #tm.train()  # 开始训练模型
    tm.test_images()  # 识别图片示例


if __name__ == '__main__':
    main()
