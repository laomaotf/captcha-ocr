# -*- coding: UTF-8 -*-
"""
1.从指定页面上下载一个验证码
2.发送到到识别服务，保存识别结果
"""
import datetime
import requests
from io import BytesIO
import time,os
import json
import random


with open("config.json", "r",encoding="utf8") as f:
    config = json.load(f)

config_captcha_server = config['captcha_server']
config_ocr_server = config['recognize_server']
captcha_url = "http://{host}:{port}/{page}".format(host=config_captcha_server['host'],
                                                   port=config_captcha_server['port'],
                                                   page=config_captcha_server['page'])

ocr_url = "http://{host}:{port}/{page}".format(host=config_ocr_server['host'],
                                                   port=config_ocr_server['port'],
                                                   page=config_ocr_server['page'])

config_client = config['recognize_client']
output_folder = os.path.join(config['project_root'], config_client['output_folder'])
use_local_image = config_client['use_local_image']


#从训练集中随机选择一个图作为loca image
project_root = config["project_root"]
image_folder = os.path.join(project_root,config['captcha']['image_root'])
annotation_file = os.path.join(project_root,config['captcha']['annotation_file'])
annotations = []
with open(annotation_file, 'r') as f:
    for line in f:
        items = line.strip().split(',')
        image_path = os.path.join( image_folder, items[1].strip())
        annotations.append(image_path)

def recognize_captcha(captch_url,output_folder, rec_times = 1, image_suffix=".jpg"):
    image_file_name = 'captcha.{}'.format(image_suffix)

    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36",
    }

    def get_image_data(url):
        if url.split(':')[0] == 'http': #从网上下载图片
            while True:
                try:
                    response = requests.request("GET", url, headers=headers, timeout=6)
                    if response.text:
                        break
                    else:
                        print("retry, response.text is empty")
                except Exception as ee:
                    print(ee)
            return response.content
        else:#本地二进制
            with open(url, "rb") as f:
                content = f.read()
            return content
        return None

    for index in range(rec_times):
        #1-获得图片数据
        content = get_image_data(captch_url)

        #2-图片发送到识别服务器
        s = time.time()

        files = {'image_file': (image_file_name, BytesIO(content), 'application')}
        r = requests.post(url=ocr_url, files=files)
        e = time.time()

        #3-解析识别结果 & 保存结果凸
        print("接口响应: {}".format(r.text))
        predict_text = json.loads(r.text)["value"]
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("【{}】 index:{} 耗时：{}ms 预测结果：{}".format(now_time, index, int((e-s)*1000), predict_text))

        # 保存文件
        os.makedirs(output_folder,exist_ok=True)
        img_name = "{}_{}.{}".format(predict_text, str(time.time()).replace(".", ""), image_suffix)
        path = os.path.join(output_folder, img_name)
        with open(path, "wb") as f:
            f.write(content)




if __name__ == '__main__':
    if not use_local_image:
        recognize_captcha(captcha_url,output_folder)
    else:
        print("use local image:")
        captcha_path = random.choice(annotations)
        print("use local image:",captcha_path)
        recognize_captcha(captcha_path, output_folder)
    

