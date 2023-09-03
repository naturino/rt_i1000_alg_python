import cv2
import numpy as np
import json

class  CutCntPatch:
    def __init__(self):
        pass

    def pad(self, src, size, color=255):
        # 获取原始图片尺寸
        height, width = src.shape[:2]

        # 将大于size的图片等比例缩放到该size
        if size < max(height, width):
            # 计算缩放比例
            if height > width:
                new_height = 256
                new_width = int(width * (new_height / height))
            else:
                new_width = 256
                new_height = int(height * (new_width / width))

            # 缩放图片
            src = cv2.resize(src, (new_width, new_height))

        h1, w1 = src.shape[:2]
        # 计算在image2中绘制image1的起始位置
        x_offset = int((size - w1) // 2)
        y_offset = int((size - h1) // 2)

        if len(src.shape) == 3:
            _, _, c1 = src.shape
            dst = np.ones((size, size, c1)) * color
            # 在image2上绘制image1
            dst[y_offset:y_offset + h1, x_offset:x_offset + w1, :] = src

        elif len(src.shape) == 2:
            dst = np.ones((size, size)) * color
            # 在image2上绘制image1
            dst[y_offset:y_offset + h1, x_offset:x_offset + w1] = src
        else:
            dst = src

        return dst

    def crop(self,img,cnt,isRotate=False):

        cnt = np.array(cnt, dtype=np.int64)
        obj_img = self.getCntImg(img, cnt)

        if isRotate:
            patch = self.cropRotate(obj_img, cnt)
        else:
            patch = self.cropPatch(obj_img, cnt)

        return patch

    def getCntImg(self,img,cnt):
        mask = np.zeros(img.shape, dtype=np.uint8)
        cnt = np.array(cnt, dtype=np.int64)
        cv2.drawContours(mask, [cnt], -1, (1, 1, 1), -1)
        img[img == 0] = 1
        dst = img * mask
        dst[dst == 0] = 255
        return dst

    def cropPatch(self,img, cnt):

        x, y, w, h = cv2.boundingRect(cnt)
        if len(img.shape) == 2:
            dst = img[y:y + h, x:x + w]
        else:
            dst = img[y:y + h, x:x + w,:]
        return dst

    def cropRotate(self,img, cnt):
        rect = cv2.minAreaRect(cnt)
        # get the parameter of the small rectangle
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # get row and col num in img
        height, width = img.shape[0], img.shape[1]
        # if width>height:
        #     angle=-angle
        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (width, height))

        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, size, center)
        # print('img_crop:',img_crop.shape)
        if img_crop.shape[0]<img_crop.shape[1]:
            img_crop = np.rot90(img_crop)

        return img_crop

    def getImgCnts(self,img,binary = True):

        if len(img.shape) > 2:
            # 将图像转换为灰度
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 对灰度图像进行阈值处理
        if binary:
            # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV)
            # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img,contours,-1,0,2)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        return contours

    def max_size(self, patches):
        w_h = []
        for patch in patches:
            w_h.extend(patch.shape)
        return int(max(w_h))

if __name__ == '__main__':
    json_path = "E:/adam/code/python/i1000/mmdet-3.0.0/tmp/jsons/demo2.json"
    img_path = "E:/adam/code/python/i1000/mmdet-3.0.0/assets/img/demo2.png"
    img = cv2.imread(img_path,0)
    json_file = open(json_path,'r')
    json_list = json.load(json_file)
    ccp = CutCntPatch()
    cnts = []
    for json_dict in json_list:
        cnt = json_dict["cnts"][0]
        cnts.append(cnt)
        patch = ccp.run(img,cnt,True,True)
