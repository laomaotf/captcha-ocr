# -*- coding: UTF-8 -*-
"""
模拟一个页面，每次访问会随机显示一个验证码
可以测试验证码爬虫程序。
浏览器访问方法: host:port/page/
"""
import os
import random
from PIL import Image, ImageDraw
from flask import Flask, request, Response, make_response
import json
import io
import numpy as np


# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


with open("config.json", "r",encoding='utf8') as f:
    config = json.load(f)


#简单起见，随机从验证码训练集中选择一个图显示，不再动态生成
project_root = config["project_root"]
image_folder = os.path.join(project_root,config['captcha']['image_root'])
annotation_file = os.path.join(project_root,config['captcha']['annotation_file'])

port = config['captcha_server']['port']
host = config['captcha_server']['host']

annotations = []
with open(annotation_file, 'r') as f:
    for line in f:
        items = line.strip().split(',')
        image_path = os.path.join( image_folder, items[1].strip())
        annotations.append(image_path)


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def gen_special_img():
    path = random.choice(annotations)
    img = Image.open(path)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


@app.route('/{}'.format(config['captcha_server']['page']), methods=['GET'])
def show_photo():
    if request.method == 'GET':
        image_data = gen_special_img()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        pass


if __name__ == '__main__':
    app.run(
        host=host,
        port=port,
        debug=True
    )
