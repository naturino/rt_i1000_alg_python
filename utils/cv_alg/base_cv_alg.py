import cv2
import numpy as np

class BasicCVAlg:
    def __init__(self):
        self.cntsAlg = BasicCVContoursAlg()

class BasicCVContoursAlg:
    def __init__(self):
        pass

    def findContoursByImage(self,src,binary):
        img = src

        if img is None:
            return []

        if len(img.shape) > 2:
            # 将图像转换为灰度
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 对灰度图像进行阈值处理
        if binary:
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def filterByArea(self,cnts,min_area,max_area):

        if min_area > max_area:
            return cnts

        dst = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > min_area and area < max_area:
                dst.append(cnt)
        return dst

    def maskMean(self,src,mask):

        mask = mask.astype(np.uint8)
        masked_image = cv2.bitwise_and(src, mask, mask=mask)
        mean_color = cv2.mean(masked_image)
        if len(src.shape) == 2:
            mean_color = mean_color[0]
        return mean_color

    def filterByColor(self,src,cnts,min_color,max_color):

        if min_color > max_color:
            return cnts

        dst = []
        for cnt in cnts:
            mask = np.zeros(src.shape)
            cv2.drawContours(mask,[cnt],-1,255,-1)
            mean_color = self.maskMean(src,mask)
            if mean_color > min_color and mean_color < max_color:
                dst.append(cnt)
                print(min_color,max_color,mean_color)

        return dst