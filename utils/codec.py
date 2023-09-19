import cv2
import numpy as np
import json
import base64

class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            png_bytes = cv2.imencode('.png', obj)[1].tobytes()
            base64_string = base64.b64encode(png_bytes).decode('utf-8') # 使用Base64编码将字节对象转换为字符串
            return base64_string
        else:
            return super(DataEncoder, self).default(obj)

class DataTransform:
    def bytes2mat(self, img):
        '''二进制图片转cv2

        :param im: 二进制图片数据，bytes
        :return: cv2图像，numpy.ndarray
        '''
        return cv2.imdecode(np.array(bytearray(img), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取


    def mat2bytes(self, bytes):
        '''cv2转二进制图片

        :param im: cv2图像，numpy.ndarray
        :return: 二进制图片数据，bytes
        '''
        return np.array(cv2.imencode('.png', bytes)[1]).tobytes()
