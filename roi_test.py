import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 在ROI区域分割出目标物料图像
def image_pre(image_source):
    # 图像处理：灰度化
    dst = cv2.cvtColor(image_source,cv2.COLOR_RGB2GRAY)
    # 图像处理：二值化（可适当修改阈值范围）
    ret,dst_2=cv2.threshold(dst,100,230,cv2.THRESH_BINARY)
    cv2.imshow("image", dst_2)
    cv2.waitKey(0)
    # 图像处理：形态学开运算
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(dst_2, cv2.MORPH_OPEN, kernel)
    # 图像处理：包络轮廓
    # binary, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, binary = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 如果找不到轮廓，则返回原图，防止空载报错
    src_roi = image_source
    for i in range(0,len(contours)):
        # 打印轮廓面积
        area = cv2.contourArea(contours[i])
        print(area)
        # 如果提取的轮廓面积少于50000，跳过，继续寻找轮廓
        if area<50000:
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
        cv2.imshow("image", src_roi)
        cv2.waitKey(0)
        cv2.imwrite('./image/img.jpg', src_roi)
        break
    return src_roi

# 运行主函数
if __name__ == "__main__":
        path = 'D:/VisionMaster/image.jpg'
        img = cv2.imread(path,1)
        # 裁剪坐标为[y0:y1, x0:x1]
        image_source = img[445:1373, 877:1595]
        cv2.imwrite('./image/roi.jpg',image_source)
        crop_img = image_pre(image_source)
