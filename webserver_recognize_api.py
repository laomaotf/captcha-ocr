# -*- coding: UTF-8 -*-
"""
模拟验证码识别页面，接收一个验证码图片，返回识别结果
访问url
host:port/page/
"""
import json
from io import BytesIO
import os
#from task_exports import CLASS_SEGMENT_OCR as CAPTCHA_API
from task_exports import CLASS_TOY as CAPTCHA_API

import time
from flask import Flask, request, jsonify, Response
from PIL import Image

# 默认使用CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with open("config.json", "r",encoding="utf8") as f:
    conf = json.load(f)

output_folder = os.path.join(conf['project_root'],conf['recognize_server']['output_folder'])

# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# 生成识别对象
R = CAPTCHA_API("config.json")


def response_headers(content): #失败是显示的内容
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/{}'.format(conf['recognize_server']['page']), methods=['POST'])
def up_image():
    if request.method == 'POST' and request.files.get('image_file'):
        timec = str(time.time()).replace(".", "")
        file = request.files.get('image_file')
        img = file.read()
        img = BytesIO(img)
        img = Image.open(img, mode="r")
        # username = request.form.get("name")
        print("接收图片尺寸: {}".format(img.size))
        s = time.time()
        value = R.run(img)
        e = time.time()
        print("识别结果: {}".format(value))
        # 保存图片
        os.makedirs(output_folder,exist_ok=True)
        print("保存图片： {}{}_{}.{}".format(output_folder, value, timec, ".jpg"))
        file_name = "{}_{}.{}".format(value, timec, ".jpg")
        file_path = os.path.join(output_folder + file_name)
        img.save(file_path)
        result = {
            'time': timec,   # 时间戳
            'value': value,  # 预测的结果
            'speed_time(ms)': int((e - s) * 1000)  # 识别耗费的时间
        }
        img.close()
        return jsonify(result)
    else:
        content = json.dumps({"error_code": "1001"})
        resp = response_headers(content)
        return resp


if __name__ == '__main__':
    app.run(
        host=conf['recognize_server']['host'],
        port=conf['recognize_server']['port'],
        debug=True  #调试模式
    )
