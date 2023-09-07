import cv2
import numpy as np

class Registration:
    def __init__(self):
        self.method = ['ORB', 'SIFT']

    def drawKeyPoints(self,image,keypoints,show = False):
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0), flags=0)
        if show:
            cv2.imshow('Image2 with Keypoints', image_with_keypoints)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image_with_keypoints

    def drawMatches(self,image1,image2,keypoints1,keypoints2,good_matches,show=True):
        match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if show:
            cv2.imshow('Match_image', match_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return match_image

    def preprocess(self,image1,image2):
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        size = (max(h1,h2),max(w1,w2))
        image1 = self.resize_and_pad(image1, size,255)
        image2 = self.resize_and_pad(image2, size,255)
        return image1,image2

    def resize_and_pad(self, image, target_size, pad_value):

        # 获取原始图像尺寸和目标尺寸
        original_h, original_w = image.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例
        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # 缩放图像
        resized_image = cv2.resize(image, (new_w, new_h))

        # 创建一个填充后的图像
        if len(image.shape) == 3:
            channels = image.shape[-1]
            padded_image = np.full((target_h, target_w, channels), pad_value, dtype=np.uint8)
        else:
            padded_image = np.full((target_h, target_w), pad_value, dtype=np.uint8)
        # 将缩放后的图像放置在中心，剩余部分填充为指定值
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        return padded_image

    def run(self,image1,image2,method):

        if method not in self.method:
            return

        idx = self.method.index(method)

        if idx == 0:
            self.ORB(image1,image2, 1000)

        elif idx == 1:
            self.SIFT(image1,image2)

    def ORB(self, image1, image2, nfeatures = 500):

        # 创建ORB检测器
        orb = cv2.ORB_create(nfeatures=nfeatures)

        # 在两张图片中找到特征点和描述子
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)

        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 找到最佳匹配
        matches = bf.match(descriptors1, descriptors2)

        # 将匹配按照距离进行排序
        matches = sorted(matches, key=lambda x: x.distance)

        # 取前N个匹配，N可以根据需要进行调整
        N = 1
        good_matches = matches[:N]
        self.drawMatches(image1, image2, keypoints1, keypoints2, good_matches, show=True)

        # 提取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算透视变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 应用透视变换
        height, width = image1.shape
        registered_image = cv2.warpPerspective(image1, M, (width, height))

        return registered_image

    def SIFT(self, image1, image2, score = 0.5):
        # 创建SIFT特征检测器
        sift = cv2.SIFT_create()

        # 检测特征点和计算特征描述子
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        # 使用FLANN匹配器进行特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # 筛选好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < score * n.distance:
                good_matches.append(m)

        self.drawMatches(image1, image2, keypoints1, keypoints2, good_matches, show=True)

        # 提取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用RANSAC算法估计变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 应用透视变换
        height, width = image1.shape
        registered_image = cv2.warpPerspective(image1, M, (width, height))

        return registered_image


if __name__ == '__main__':
    root_path = 'D:/Users/Desktop/re'
    my_registration = Registration()

    # 读取两张图片
    image1 = cv2.imread(f'{root_path}/enhance.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(f'{root_path}/3.jpg', cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1,(1024,1024))
    registered_image = my_registration.run(image1,image2,'SIFT')

    # 保存配准后的图片
    # cv2.imwrite(f'{root_path}/registered_image.jpg', registered_image)

    # 显示配准后的图片
    cv2.imshow('Registered Image', registered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
