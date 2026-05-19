import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn
import time
import socket

# 创建一个TCP客户端
socket_client = socket.socket()
host = "127.0.0.1"
port = 7890
socket_client.connect((host,port))

# 模型推理
def modelpre(src_roi):
    grey_img = cv2.cvtColor(src_roi, cv2.COLOR_BGR2GRAY)
    tsfrm = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 0.水果 1.蔬菜 2.服装 3.零食
    classes = ('水果','蔬菜','服装','零食')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_eval = models.resnet18(pretrained=False)
    num_ftrs = model_eval.fc.in_features
    model_eval.fc = nn.Linear(num_ftrs, 4)
    model_eval.load_state_dict(torch.load('model\Model-20230319.pkl', map_location=device))
    # 在推理前，务必调用model.eval()去设置dropout和batch normalization层为评估模式
    model_eval.eval()
    # OpenCV转PIL格式
    image = Image.fromarray(cv2.cvtColor(grey_img, cv2.COLOR_GRAY2RGB))
    # PIL图像数据转换为tensor数据，并归一化
    img = tsfrm(image)
    # 图像增加1维[batch_size,通道,高,宽]
    img = img.unsqueeze(0)
    # 输出推理结果
    output = model_eval(img)
    # prob是4个分类的概率
    prob = str(F.softmax(output, dim=1))
    print(prob)
    value, predicted = torch.max(output.data, 1)
    prob_stat = prob[prob.find("[[")+2 : prob.find("]]")].replace(" ", "").split(",")
    if float(prob_stat[0])>0.5 or float(prob_stat[1])>0.9 or float(prob_stat[2])>0.99 or float(prob_stat[3])>0.99:
        label = predicted.numpy()[0]
        pred_class = classes[predicted.item()]
    else:
        label = 4
        pred_class = '未知物料'
    return label,pred_class

# 图像预处理
def image_pre(image_source):
    # 图像处理：灰度化
    dst = cv2.cvtColor(image_source,cv2.COLOR_RGB2GRAY)
    # 图像处理：二值化（可适当修改阈值范围）
    ret,dst_2=cv2.threshold(dst,100,230,cv2.THRESH_BINARY)
    # 图像处理：形态学开运算
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(dst_2, cv2.MORPH_OPEN, kernel)
    # 图像处理：包络轮廓
    contours, binary = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 如果找不到轮廓，则返回原图，防止空载报错
    src_roi = image_source
    for i in range(0,len(contours)):
        # 打印轮廓面积
        area = cv2.contourArea(contours[i])
        # print(area)
        # 如果提取的轮廓面积少于50000，跳过，继续寻找轮廓
        if area < 50000:
            continue
        # 左上角坐标(x_l,y_l)、宽、高
        x_l, y_l, w, h = cv2.boundingRect(contours[i])
        # 中心点坐标
        x = (x_l+h)/2;
        y = (y_l+w)/2;
        # 获取所有轮廓（Blob连通区）最小外接矩形的（中心(x,y), (宽,高), 旋转角度）数据对象rect
        rect = cv2.minAreaRect(contours[i])
        # 获取最小外接矩形的4个顶点坐标(左上角、左下角、右下角、右上角)
        box = cv2.boxPoints(rect)
        # 数据类型转换
        box = np.int64(box)
        # 根据最小外接矩形的中心坐标与角度，构建一个旋转矩阵rot_img
        # 输入目标轮廓矩形中心点坐标center = rect[0]，矩形角度angle = rect[2]
        rot_img = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
        # 使用前面获得的四个矩形顶点坐标数组[box]，在原图层绘制轮廓
        cv2.drawContours(image_source, [box], 0, (0, 255, 0), 1)   
        # 利用旋转矩阵rot_img,原图层img实现中心仿射变换，变换后的图层尺寸保持不变
        # height = image_source.shape[0]，width = image_source.shape[1]
        img_waf = cv2.warpAffine(image_source, rot_img, (image_source.shape[0],image_source.shape[1]))
        # 仿射变换后得到img_waf图层中心点坐标和目标轮廓最小外接矩形中心点一致,
        # 根据数据对象rect提取出矩形中心点坐标和宽高x，y，w，h，分别得到矩形行(x)和列(y)的起始点和结束点
        # 在img_waf中剪裁出纠正角度后的目标图层src_roi
        # 语句原型(img_waf[y-int(h/2): y+int(h/2)+4, x-int(w/2): x+int(w/2)+4])
        src_roi = img_waf[int(rect[0][1])-int((rect[1][1])/2)+4:int(rect[0][1])+int((rect[1][1])/2)-4,
                        int(rect[0][0])-int((rect[1][0])/2)+4:int(rect[0][0])+int((rect[1][0])/2)-4]
        cv2.imwrite('./image/img.jpg', src_roi)
        break
    return src_roi

# 显示推理结果
def showResult(img,text):
    # 设置图片显示分辨率
    newimg = cv2.resize(img,(640,480))
    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, 48)  
    img_pil = Image.fromarray(newimg)
    draw = ImageDraw.Draw(img_pil)
    # 绘制文字信息
    draw.text((10, 100), text, font=font, fill=(0, 0, 255))
    bk_img = np.array(img_pil)
    # 获取实时时间作为命名，将识别图片和结果保存到文件夹
    str_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    img_name = str_time + '.jpg'
    cv2.imwrite('./result/' + img_name, bk_img)    
    cv2.imshow('frame', bk_img)
    # 持续显示3s
    key = cv2.waitKey(3000)
    cv2.destroyAllWindows()

# ROI区域裁剪函数
def image_roi():
    image_source = cv2.imread('image/nj.jpg')
    if image_source is None:
        print('Image is not found.')
        return ''
    # 裁剪坐标为[y0:y1, x0:x1]
    # ROI区域(用户根据需求修改)
    img_roi = image_source[445:1373, 877:1595]
    # 保存图像
    cv2.imwrite('./image/roi.jpg',img_roi)
    return img_roi

# 清空缓存图片
def remove_result():
    path = './result/'
    # 判断是否存在result文件夹
    if os.path.exists(path):                  
        for i in os.listdir(path):
            # 遍历拼接文件路径
            path_file = os.path.join(path, i) 
            # 判断该路径对象是否为文件
            if os.path.isfile(path_file):
                # 删掉文件    
                os.remove(path_file)          
    else:
        # 新建一个result文件夹
        os.mkdir(path)                        

# 主程序入口
if __name__ == "__main__":
    # 每次运行前清空上次运行识别结果缓存图片
    remove_result()    
    print('Waiting...')
    while True:
        data = socket_client.recv(1024).decode('utf-8')
        print(data)
        msg = ''
        if data == 'ok':
            # 方案一：原图进行ROI裁剪
            # img_roi = image_roi()
            # # 将裁剪后的图像进行预处理
            # src_roi = image_pre(img_roi)
            
            # 方案二 仿射变换方案
            src_roi = cv2.imread('image/nj.jpg')

            # 将图像预处理后的图像进行模型推理
            label, pred_class = modelpre(src_roi)
            # 打印推理结果
            print(label, pred_class)
            # 显示推理结果
            showResult(src_roi, pred_class)
            if label == 0:
                msg = 'A'
            elif label == 1:
                msg = 'B'
            elif label == 2:
                msg = 'C'
            elif label == 3:
                msg = 'D'
            elif label == 4:
                msg = 'error'
            socket_client.send(msg.encode())
            data = ''
        else:
            msg = 'null'
            socket_client.send(msg.encode())
            print('data error!')
         

