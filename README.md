# 1. 参考
本工程用来学习tensorflow2，代码主要参考以下项目  
[cnn_captcha](https://github.com/nickliqian/cnn_captcha)  
[yymnist](https://github.com/YunYang1994/yymnist)

 

# 2. 代码说明
## 2.1 配置文件
工程配置文件只有一个config.json, 不同任务对应不同的json字段，
具体参见config.json的注释
* 所有路径都是基于工程根路径
* charset::chars的次序就是索引，内部不对labels做排序

## 2.2 验证码图片生成
* 功能: 参考yymnist思路生成简单的验证码图片，方便实验。
* 配置: config::captcha字段
* 脚本: gen_images.py

## 2.3 toy task
* 功能: 验证码图片 -> CNN -> 验证码
* 配置: config::task_toy字段
* 脚本: tools/task_toy_gen_tfrec.py, task_toy_train.py
* 部署: task_exports.CLASS_TOY

## 2.4 字符分割 [UNET]
* 功能: 验证码图片 -> UNET -> 字符mask -> 字符bbox
* 配置: config::task_segment字段
* 脚本: tools/task_segment_gen_tfrec.py, task_segment_train.py
  
## 2.5 单字符OCR
* 功能: 字符图片 -> CNN -> 字符
* 配置: config::task_ocr
* 脚本: tools/task_ocr_gen_tfrec.py, task_ocr_train.py


## 2.6字符分割 +单字符OCR
* 功能: 组合两个子任务，完成验证码识别。字符图片 -> 字符分割 -> 单字符OCR -> 验证码
* 部署: task_exports.CLASS_SEGMENT_OCR

## 2.7 字符检测 [简化CenterNet]
* 功能: 字符图片 -> CNN -> 字符框
* 配置: config::task_detect
* 脚本: tools/task_detect_gen_tfrec.py, task_detect_train.py

# 3. web端部署
## 3.1 模拟验证码生成服务
* 功能: 生成一个page，随机显示一个验证码图片
* 配置: config::captcha_server

## 3.2 模拟验证码识别服务
* 功能：搭建服务，接受客户端发送的验证码图，返回识别结果
* 配置: config::recognize_server

## 3.3 验证码识别客户端
* 功能：从本地或远端得到验证码图片，发送给识别服务，保存识别结果
* 配置: recognize_client
