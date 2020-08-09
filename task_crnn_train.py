# -*- coding: utf-8 -*-


import tensorflow as tf
tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os,shutil
from networks.crnn import get_crnn
from datetime import datetime
import json
################################################
#参考 https://github.com/FLming/CRNN.tf2



class CTCLoss(keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1,
                 reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major #False indicates logist shape following (batch, time, logist)
        self.blank_index = blank_index #BLANK在vocab中的索引 = self.blank_index + num_classes。
                                       #blank_index=-1，说明字典中blank对应的索引值是num_classes - 1,即vocab中最后一个字符
                                       #是blank

    def call(self, y_true, y_pred):
        if not isinstance(y_true,tf.sparse.SparseTensor):
            return 0.0
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.reduce_mean(loss)


class WordAccuracy(keras.metrics.Metric):
    """
    Calculate the word accuracy between y_true and y_pred.
    """

    def __init__(self, name='word_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not isinstance(y_true,tf.sparse.SparseTensor):
            return
        batch_size = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_true = self.to_dense(y_true, [batch_size, max_width])
        y_pred = self.to_dense(decoded[0], [batch_size, max_width])
        num_errors = tf.math.reduce_any(
            tf.math.not_equal(y_true, y_pred), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.reduce_sum(num_errors)
        batch_size = tf.cast(batch_size, tf.float32)
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def to_dense(self, tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)

class TrainModel:
    def __init__(self, config, config_charset, config_captcha, config_task,verify=False):
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


        os.makedirs(self.output_log_folder, exist_ok=True)

        self.num_train_image = config_captcha['num_train_image']
        self.num_test_image = config_captcha['num_image'] -config_captcha['num_train_image']

        # 相关信息打印
        self.max_captcha = config_captcha['max_text_length']
        self.chars = config_charset['chars']
        self.char_set_len = len(config_charset['chars'])
        print("-->图片尺寸: {} X {}".format(self.image_height, self.image_width))
        print("-->验证码最大长度: {}".format(self.max_captcha))
        print("-->验证码共{}类".format(self.char_set_len))
        print("-->使用测试集为 {}".format(config_task['train_tfrec_file']))
        print("-->使验证集为 {}".format(config_task['test_tfrec_file']))


    def load_train_test(self):
        # 把读取的example解析成的字典
        def _parse_function(example_proto):
            features = {'data': tf.io.FixedLenFeature((self.image_height, self.image_width, self.image_channel), tf.int64),
                        #'label': tf.io.FixedLenFeature([self.max_captcha * self.char_set_len], tf.int64)}
                        'label': tf.io.VarLenFeature(tf.int64)}
            parsed_features = tf.io.parse_single_example(example_proto, features)
            parsed_features['data'] = tf.cast(parsed_features['data'], tf.uint8)
            #CTC Loss实现中要求gt是sparse
            #parsed_features['label'] = tf.sparse.to_dense(parsed_features['label'], default_value=0)

            if not isinstance(parsed_features['label'], tf.sparse.SparseTensor):
                print("not sparser")

            return parsed_features

        dataset_train = tf.data.TFRecordDataset(self.train_tfrec_file)
        dataset_test = tf.data.TFRecordDataset(self.test_tfrec_file)

        dataset_train = dataset_train.map(_parse_function)
        dataset_test = dataset_test.map(_parse_function)

        #dataset_test.repeat()
        #dataset_train.repeat()
        return dataset_train, dataset_test


    def train_cnn(self):

        os.makedirs(self.output_model_folder, exist_ok=False)

        ##########################################
        #tensorboard的value单独放一个目录
        file_writer = tf.summary.create_file_writer( os.path.join(self.output_log_folder, "params") )
        file_writer.set_as_default()

        train_dataset, test_dataset = self.load_train_test()

        model = get_crnn(input_shape=(self.image_height, self.image_width, self.image_channel),
                         num_classes=self.char_set_len)


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
            image = tf.image.random_contrast(image,lower=0.8, upper=1.3)  # Random brightness

            image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness

            image = tf.image.random_saturation(image,lower=0.8,upper=1.3)


            image = tf.image.random_hue(image, max_delta=0.5)  # Random hue


            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]


            return (image, label)

        # dataset转换成batches
        # 这里调用函数把tfrecord转换成(data,label)二元组，作为后续fit的输入
        train_batches = (
            train_dataset
                .shuffle(self.num_train_image )
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

        # 连续5个epoch loss的变化小于1e-2 earlystop
        #callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]

        ###########################################################################################
        optimizer = tf.keras.optimizers.Adam(1e-4)
        calc_model_loss = CTCLoss()
        record_model_accuracy = WordAccuracy()
        iter_total = 0
        for current_epoch in range(self.epoch_stop):
            loss_in_epoch = 0
            sample_in_epoch = 0
            record_model_accuracy.reset_states()
            epoch_t0 = time.time()
            for iter_in_epoch, batch_train in enumerate(train_batches):
                iter_total += 1
                data, label_sparse = batch_train
                with tf.GradientTape() as tape:
                    pred_onehot = model(data)
                    loss = calc_model_loss(label_sparse, pred_onehot)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                loss_in_epoch += loss
                sample_in_epoch += self.train_batch_size

                record_model_accuracy.update_state(label_sparse,pred_onehot)

                if 0 == (iter_total % 100):
                    print("Iteration {}, Loss {:.5f}, Speed {:.3f} it/s".format(iter_total,
                                                                    loss_in_epoch * 1.0 / sample_in_epoch,
                                                                    sample_in_epoch / (time.time() - epoch_t0)))

            print("Epoch {}, Loss {:.5f}, Time {:.3f} hours, accuracy {:.3f}".format(current_epoch,
                                                            loss_in_epoch * 1.0 / sample_in_epoch,
                                                            (time.time() - epoch_t0) / 3600.0,
                                                            record_model_accuracy.result()))

            tf.summary.scalar('TrainLoss', data=loss_in_epoch * 1.0 / sample_in_epoch, step=current_epoch)
            tf.summary.scalar('TrainAcc', data=record_model_accuracy.result(), step=current_epoch)

            record_model_accuracy.reset_states()
            loss_in_epoch = 0
            sample_in_epoch = 0
            for iter_in_epoch, batch_train in enumerate(test_batches):
                data, label_sparse = batch_train
                pred_onehot = model.predict(data)
                loss = calc_model_loss(label_sparse, pred_onehot)
                loss_in_epoch += loss
                sample_in_epoch += self.train_batch_size
                record_model_accuracy.update_state(label_sparse, pred_onehot)

            print("\tTest Loss {:.5f}  accuracy {:.3f}".format(
                loss_in_epoch * 1.0 / sample_in_epoch,
                record_model_accuracy.result()))

            tf.summary.scalar('TestLoss', data=loss_in_epoch * 1.0 / sample_in_epoch, step=current_epoch)
            tf.summary.scalar('TestAcc', data=record_model_accuracy.result(), step=current_epoch)

            model.save_weights(ckpt_path.format(epoch=current_epoch))

        # # profile_batch of TensorBoard() will make training in eager model(slowing training), set zero to disable it
        # callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, save_freq="epoch"),
        #              tf.keras.callbacks.TensorBoard(self.output_log_folder, update_freq="epoch"),
        #              #tf.keras.callbacks.LearningRateScheduler(current_lr),
        #              ]
        # try:
        #     #print(len( self.train_images_list) // self.train_batch_size)
        #     history = model.fit(train_batches, epochs=self.epoch_stop ,
        #                         validation_data=test_batches, #valid data没设设置steps_per_epoch!!!
        #                         #steps_per_epoch = len(self.train_images_list) // self.train_batch_size, #配合repeat使用
        #                         #steps_per_epoch=1,
        #                         verbose=False,callbacks=callbacks)
        # except KeyboardInterrupt:
        #     model.save_weights(ckpt_path.format(epoch=self.epoch_stop + 1))
        #     print("save model before quit!")
        # model.save_weights(ckpt_path.format(epoch=self.epoch_stop + 1))


        #
        # acc = history.history['word_accuracy']
        # val_acc = history.history['val_word_accuracy']
        #
        # loss = history.history['ctc_loss']
        # val_loss = history.history['val_ctc_loss']
        #
        # epochs_range = range(self.epoch_stop)
        # plt.figure(figsize=(8, 8))
        # plt.subplot(1, 2, 1)
        # plt.plot(epochs_range, acc, label='Training Accuracy')
        # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        # plt.legend(loc='lower right')
        # plt.title('Training and Validation Accuracy')
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(epochs_range, loss, label='Training Loss')
        # plt.plot(epochs_range, val_loss, label='Validation Loss')
        # plt.legend(loc='upper right')
        # plt.title('Training and Validation Loss')
        # plt.show()


    def test_images(self):
        def _list2sparse(value, dense_shape):
            assert len(dense_shape) == 2 and dense_shape[0] == 1
            indices = []
            for ind,val in enumerate(value):
                indices.append((0,ind))
            return tf.sparse.SparseTensor(indices=indices,values=value, dense_shape=dense_shape)

        def _to_dense(tensor, shape):
            tensor = tf.sparse.reset_shape(tensor, shape)
            tensor = tf.sparse.to_dense(tensor, default_value=-1)
            tensor = tf.cast(tensor, tf.float32)
            return tensor

        model = get_crnn(input_shape=(self.image_height, self.image_width, self.image_channel),
                         num_classes=self.char_set_len)

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
            gt_label = ""
            y_true = []
            for object in objects:
                gt_label += self.chars[object[-1]]
                y_true.append(object[-1])

            captcha_array = Image.open(image_path).resize((self.image_width,self.image_height),Image.BILINEAR)
            captcha_array = np.asarray(captcha_array,dtype=np.uint8)

            axes[k].imshow(captcha_array)

            data = tf.image.convert_image_dtype(captcha_array, tf.float32)
            data = tf.reshape(data,(1,self.image_height,self.image_width,-1))

            y_pred = model.predict(data)

            batch_size = 1
            max_width = self.max_captcha
            logit_length = tf.fill([1], tf.shape(y_pred)[1])
            decoded, _ = tf.nn.ctc_greedy_decoder(
                inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                sequence_length=logit_length)
            y_true = _to_dense(_list2sparse(y_true,[batch_size, max_width]), [batch_size, max_width])
            y_pred = _to_dense(decoded[0], [batch_size, max_width])
            num_errors = tf.math.reduce_any(
                tf.math.not_equal(y_true, y_pred), axis=1)
            num_errors = tf.cast(num_errors, tf.float32)
            if num_errors > 0:
                text_pred = ""
                for y in y_pred.numpy()[0].tolist():
                    if y < 0:
                        continue
                    y = int(y)
                    text_pred += self.chars[y]
                axes[k].set_title("error: {} -> {}".format(gt_label, text_pred))
            else:
                axes[k].set_title("{}".format(gt_label))

        plt.show()

#############################################################

#############################################################
def main():
    # import numpy as np
    #
    # y_true = np.array([[4, 2, 1], [2, 3, 0]])  # (2, 3)
    # y_pred = keras.utils.to_categorical(np.array([[4, 1, 3], [1, 2, 4]]), 6)  # (2, 3, 5)
    #
    # input_length = np.array([[3], [3]])  # (2, 1)
    # label_length = np.array([[3], [3]])  # (2, 1)
    #
    # cost = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    #
    # return
    with open("config.json", "r",encoding='utf8') as f:
        config = json.load(f)
    config_captcha = config['captcha']
    config_charset = config['charset']
    config_captcha['image_root'] = os.path.join(config['project_root'], config_captcha['image_root'])
    config_captcha['annotation_file'] = os.path.join(config['project_root'], config_captcha['annotation_file'])

    config_task = config['task_crnn']
    if config_captcha['max_text_length'] != config_captcha['min_text_length']:
        config_charset['chars'] += [' ']
    config_task['train_tfrec_file'] = os.path.join(config['project_root'], config_task['train_tfrec_file'])
    config_task['test_tfrec_file'] = os.path.join(config['project_root'], config_task['test_tfrec_file'])
    config_task['output_folder'] = os.path.join(config['project_root'], config_task['output_folder'])


    tm = TrainModel(config, config_charset, config_captcha, config_task, verify=False)
    #tm.train_cnn()  # 开始训练模型
    tm.test_images()  # 识别图片示例


if __name__ == '__main__':
    main()
