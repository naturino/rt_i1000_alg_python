from i1000_alg import *
import loger
import os
import rt_socket
import torch
import threading
from mmengine.fileio import dump
from model import FileOperate


class DeepCell():
    def __init__(self, config_path):
        # 读取配置文件
        config = self.__read_config(config_path)

        # 初始化配置文件
        config_json = dump(config, file_format='json', indent=4)
        temp_path = config['TEMP_PATH']
        if config['TEMP_PATH'] == '':
            current_path = os.getcwd()
            config['TEMP_PATH'] = os.path.join(current_path, 'alg', 'temp').replace('\\', '/')
            temp_path = config['TEMP_PATH']
        self.file_operate = FileOperate()
        self.file_operate.del_dir(temp_path)

        # 初始化日志
        self.logger = loger.Logger('alg/log')
        self.logger.info(f"Config Info: {config_json}")
        self.logger.info(f"DeepCell Start")

        gpu_info = self.__get_gpu_info()
        self.logger.info(f"Find GPU: {gpu_info}")

        # 初始化变量
        self.empty = config['EMPTY']
        self.file_type = config['IMAGE_TYPE']

        # 创建一个锁对象
        self.run_lock = threading.Lock()

        # 启动模型
        self.start_model(config)

        # 开启通信
        self.start_socket()

    def __read_config(self, path):
        dst = {}
        exec(open(path).read(), dst)
        del_key = '__builtins__'
        if del_key in dst.keys():
            del dst[del_key]
        return dst

    def __get_gpu_info(self):
        gpu_info = ''
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            gpu_info = gpu.name
        return gpu_info

    def model_test(self):
        self.run('alg/assets/img/demo1/1_7.png*0')
        self.run('alg/assets/img/demo1/1_28.png*1')
        self.run('alg/assets/img/demo3.png*2')
        self.run('alg/assets/img/demo2.jpg')
        self.logger.info(f"Model Test Successful")

    def model_debug(self):
        floder = 'F:/adam/i1000/adk/adk_unet'
        for root, _, file_list in os.walk(floder):
            for file in file_list:
                path = os.path.join(root, file) + '*2'
                self.run(path)

    def start_model(self,config):

        self.task_model = {}

        self.jsons_path = config['JSONS_PATH']
        self.only_analyse_json = False

        if self.jsons_path != '':
            self.only_analyse_json = True

        self.test = config['TEST_INIT']

        self.debug = config['DEBUG']

        if self.only_analyse_json:
            self.task_model['100XJson'] = I1000AnalyseLabelmeJsons(config, self.logger)
        else:
            # 初始化算法模型
            self.task_model['100XFlip'] = I1000FlipModel(config, self.logger)
            self.task_model['100XCls'] = I1000ClsModel(config, self.logger, self.task_model['100XFlip'])
            self.task_model['10XDet'] = I1000DetModel(config, self.logger)
            self.task_model['100XInstSeg'] = I1000InstSegModel(config, self.logger)
            self.task_model['100XAnalyse'] = I1000AnalyseModel(config, self.logger, self.task_model['100XInstSeg'],
                                                               self.task_model['100XCls'], self.task_model['100XFlip'])
            self.logger.info(f"Model Load Successful")

        # 测试模型
        if self.test:
            self.model_test()

        if self.debug:
            self.model_debug()

    def start_socket(self,config):
        self.host = config['HOST']
        self.port1 = config['PORT1']
        self.port2 = config['PORT2']

        # 启动socket1
        self.server1 = rt_socket.SocketServer(self.host, self.port1, self.logger)
        self.logger.info(f"SocketServer Init {self.host}:{self.port1} Successful")

        # 启动socket2
        self.server2 = rt_socket.SocketServer(self.host, self.port2, self.logger)
        self.logger.info(f"SocketServer Init {self.host}:{self.port2} Successful")

        # 创建两个线程来分别启动这两个服务
        thread1 = threading.Thread(target=self.server1.start, args=(self.run,))
        thread2 = threading.Thread(target=self.server2.start, args=(self.run,))

        # 启动线程
        thread1.start()
        thread2.start()

        try:
            # 在主线程中等待两个线程完成
            thread1.join()
            thread2.join()
        except KeyboardInterrupt:

            self.server1.stop()
            self.server2.stop()

    def get_task(self,info):
        dst = ''
        info_list = info.split('*')
        image_path = info_list[0]

        if len(info_list) == 2:
            img_path, task = info_list
            if task == '0':
                dst = '100XFlip'

            elif task == '1':
                dst = '100XCls'

            elif task == '2':
                if self.only_analyse_json:
                    dst = '100XJson'
                else:
                    dst = '100XAnalyse'

        elif len(info_list) == 1:
            dst = '100XAnalyse'

        return dst, image_path

    def run(self, info):
        with self.run_lock:
            dst = self.empty
            task, img_path = self.get_task(info)

            if not os.path.isfile(img_path):
                self.logger.info(f'{img_path} is not file')
                return dst
            if not self.file_operate.is_type(img_path, self.file_type):
                self.logger.info(f'{img_path} is not in {self.file_type}')
                return dst
            if task not in self.task_model.keys():
                self.logger.info(f"Dont't find {task} model")
                return dst

            dst = self.task_model[task].run(img_path)

            if type(dst) != str:
                dst = json.dumps(dst, indent=4)

            return dst

if __name__ == "__main__":
    config_path = 'alg/assets/config.py'
    server = DeepCell(config_path)
