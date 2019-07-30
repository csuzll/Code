# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd

# 图像处理

# 处理abnormal里的左右黑边的原始图
def HorizontalImgCrop(src):
    # 第1步：加载灰度图原图
    srcImg = src.copy()
    srcImg_shape = srcImg.shape  # 获取原图尺寸
    srcImg_h = srcImg_shape[0]  # 原height
    srcImg_w = srcImg_shape[1]  # 原width

    # 第2步：原图去掉左右黑边
    cropBlackBorderImg = srcImg[:, 50:(srcImg_w - 51)]

    # 第3步：高斯滤波，去除图像噪声
    blurImg = cv2.GaussianBlur(cropBlackBorderImg, (3, 3), 0)

    # 第4步：固定阈值二值化，获得二值图，此时为白底黑物
    (_, threshImg) = cv2.threshold(blurImg, 190, 255, cv2.THRESH_BINARY)

    # 第5步：反色,变为黑底白物，因为findContours()找轮廓只能在黑底白物中找
    invImg = cv2.bitwise_not(threshImg)

    # 第6步：可能目标周围有小白点，进行4次形态学腐蚀和膨胀
    closedImg = cv2.erode(invImg, None, iterations=4)
    closedImg = cv2.dilate(closedImg, None, iterations=4)

    # 第7步：找出图像上轮廓，找轮廓会修改原图
    (_, cnts, _) = cv2.findContours(closedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 第8步：找到目标图像的边界矩形
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # 取出面积最大的轮廓
    x, y, w, h = cv2.boundingRect(c)  # 找出边界矩形，（x,y）为矩形左上角的坐标，（w,h）是矩形的宽和高

    # 第9步：在剪掉黑边的图像上裁剪边界矩形
    cropImg = cropBlackBorderImg[y:(y + h), x:(x + w)]

    return cropImg

# 处理abnormal里上下黑边的原始图
def VerticalImgCrop(src):
    # 第1步：加载灰度图原图
    srcImg = src.copy()
    srcImg_shape = srcImg.shape  # 获取原图尺寸
    srcImg_h = srcImg_shape[0]  # 原height
    srcImg_w = srcImg_shape[1]  # 原width

    # 第2步：原图去掉上下黑边
    cropBlackBorderImg = srcImg[51:(srcImg_h-50), :]

    # 第3步：高斯滤波，去除图像噪声
    blurImg = cv2.GaussianBlur(cropBlackBorderImg, (3, 3), 0)

    # 第4步：固定阈值二值化，获得二值图，此时为白底黑物
    (_, threshImg) = cv2.threshold(blurImg, 190, 255, cv2.THRESH_BINARY)

    # 第5步：反色,变为黑底白物，因为findContours()找轮廓只能在黑底白物中找
    invImg = cv2.bitwise_not(threshImg)

    # 第6步：可能目标周围有小白点，进行4次形态学腐蚀和膨胀
    closedImg = cv2.erode(invImg, None, iterations=6)
    closedImg = cv2.dilate(closedImg, None, iterations=6)

    # 第7步：找出图像上轮廓，找轮廓会修改原图
    (_, cnts, _) = cv2.findContours(closedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 第8步：找到目标图像的边界矩形
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # 取出面积最大的轮廓
    x, y, w, h = cv2.boundingRect(c)  # 找出边界矩形，（x,y）为矩形左上角的坐标，（w,h）是矩形的宽和高

    # 第9步：在剪掉黑边的图像上裁剪边界矩形
    cropImg = cropBlackBorderImg[y:(y + h), x:(x + w)]

    return cropImg

# 处理normal里的左右黑边的原始图
def NormalImgCrop(src):
    # 第1步：加载灰度图原图
    srcImg = src.copy()
    srcImg_shape = srcImg.shape  # 获取原图尺寸
    srcImg_h = srcImg_shape[0]  # 原height
    srcImg_w = srcImg_shape[1]  # 原width

    # 第2步：原图去掉左右黑边
    cropBlackBorderImg = srcImg[:, 55:(srcImg_w - 55)]

    # 第3步：高斯滤波，去除图像噪声
    blurImg = cv2.GaussianBlur(cropBlackBorderImg, (3, 3), 0)

    # 第4步：固定阈值二值化，获得二值图，此时为白底黑物
    (_, threshImg) = cv2.threshold(blurImg, 190, 255, cv2.THRESH_BINARY)

    # 第5步：反色,变为黑底白物，因为findContours()找轮廓只能在黑底白物中找
    invImg = cv2.bitwise_not(threshImg)

    # 第6步：可能目标周围有小白点，进行4次形态学腐蚀和膨胀
    closedImg = cv2.erode(invImg, None, iterations=4)
    closedImg = cv2.dilate(closedImg, None, iterations=4)

    # 第7步：找出图像上轮廓，找轮廓会修改原图
    (_, cnts, _) = cv2.findContours(closedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 第8步：找到目标图像的边界矩形
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # 取出面积最大的轮廓
    x, y, w, h = cv2.boundingRect(c)  # 找出边界矩形，（x,y）为矩形左上角的坐标，（w,h）是矩形的宽和高

    # 第9步：在剪掉黑边的图像上裁剪边界矩形
    cropImg = cropBlackBorderImg[y:(y + h), x:(x + w)]

    return cropImg

# 遍历指定目录，读取，左右黑边处理，保存图像
def CropHorizontalImageDir(filepath, destpath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        srcDir = os.path.join(filepath, allDir)
        destDir = os.path.join(destpath, allDir)

        if os.path.isfile(srcDir):
            src = cv2.imread(srcDir,0)
            dst = HorizontalImgCrop(src)
            cv2.imwrite(destDir,dst)

# 遍历指定目录，读取，上下黑边处理，保存图像
def CropVerticalImageDir(filepath, destpath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        srcDir = os.path.join(filepath, allDir)
        destDir = os.path.join(destpath, allDir)

        if os.path.isfile(srcDir):
            src = cv2.imread(srcDir,0)
            dst = VerticalImgCrop(src)
            cv2.imwrite(destDir,dst)


# 遍历指定目录，读取normal，上下黑边处理，保存图像
def NormalImageDir(filepath, destpath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        srcDir = os.path.join(filepath, allDir)
        destDir = os.path.join(destpath, allDir)

        if os.path.isfile(srcDir):
            src = cv2.imread(srcDir,0)
            dst = NormalImgCrop(src)
            cv2.imwrite(destDir,dst)


if __name__ == "__main__":
    abnormalpath_1 = "../../raw_data/abnormal_new/"
    abnormalpath_2 = "../../raw_data/tempo/"
    normalpath = "../../raw_data/normal_new/"

    destpath1 = "../../ChestData/abnormal/"
    destpath2 = "../../ChestData/normal/"

    # CropHorizontalImageDir(abnormalpath_1,destpath1) # abnormal切左右
    # CropVerticalImageDir(abnormalpath_2,destpath1)   # abnormal切上下
    NormalImageDir(normalpath,destpath2)