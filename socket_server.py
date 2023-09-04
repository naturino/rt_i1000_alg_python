import queue
import socket
import select
import struct

class SocketServer:
    def __init__(self, host, port,logger):
        self.logger = logger
        self.logger.info(f"SocketServer Start")
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_connections = []
        self.message_queue = queue.Queue()
        self.is_running = False  # 用于标识服务器是否在运行
        self.decode_encode = 'utf-8'

        self.data_struct = {
            'header': {'head': 0,
                       'num': 0,
                       'cmd': 0,
                       'status': 0,
                       'encrypt': 0,
                       'len': 0},
            'data': '',
            'echo': ''}
        self.send_dst = {}

    def receive_data(self,client_socket):

        dst = self.data_struct.copy()

        # Receive head
        header_format = f"{len(self.data_struct['header'].keys())}I"
        header_size = struct.calcsize(header_format)
        header_recv = client_socket.recv(header_size)
        header_pack = struct.unpack(header_format, header_recv)
        dst['header']['head'], dst['header']['num'], dst['header']['cmd'], \
        dst['header']['status'], dst['header']['encrypt'], dst['header']['len'] = header_pack

        # Receive data
        data_format = f"{dst['header']['len']}s"
        data_recv = client_socket.recv(dst['header']['len'])
        data_pack = struct.unpack(data_format, data_recv)
        dst['data'] = data_pack[0]

        return dst

    def send_data(self, client_socket, receive_data, send_data):

        dst = receive_data
        dst['header']['len'] = len(send_data)
        dst['echo'] = send_data

        # Send head
        head_format = f"{len(dst['header'].keys())}I"
        pack_head = struct.pack(head_format, dst['header']['head'], dst['header']['num'], dst['header']['cmd'],
                                dst['header']['status'], dst['header']['encrypt'], dst['header']['len'])
        client_socket.sendall(pack_head)

        # Send echo
        data_format = f"{dst['header']['len']}s"
        pack_data = struct.pack(data_format, dst['echo'])
        client_socket.sendall(pack_data)

        self.logger.info(f'Send data successful')
        return

    def start(self,function):
        self.function = function
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.logger.info(f"Server lintening {self.host}:{self.port}")
        self.is_running = True  # 服务器正在运行

        try:
            while self.is_running:
                self._handle_connections_and_messages()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        # 停止服务器
        self.is_running = False
        # 关闭服务器套接字
        if self.server_socket:
            self.server_socket.close()

        # 关闭所有客户端连接
        for client_socket in self.client_connections:
            client_socket.close()
        self.client_connections.clear()

    def _handle_connections_and_messages(self):
        # 检查新连接
        readable, _, _ = select.select([self.server_socket], [], [], 0.1)
        for sock in readable:
            client_socket, client_address = sock.accept()
            self.logger.info(f"New connect from {client_address}")
            self.client_connections.append(client_socket)

        # 将接收到的消息放入队列进行处理
        for client_socket in self.client_connections:
            try:
                # 接收数据
                struct_data = self.receive_data(client_socket)

                if struct_data['data'] != '':
                    self.message_queue.put((client_socket, struct_data))
                else:
                    # 如果数据为空，表示客户端已关闭连接
                    client_socket.close()
                    self.client_connections.remove(client_socket)

            except Exception as e:
                self.logger.info(f"Error recv the client message: {e}")
                client_socket.close()
                self.client_connections.remove(client_socket)
                self.logger.info(f"Client close")

        # 处理队列中的消息
        while not self.message_queue.empty():

            client_socket, receive_data = self.message_queue.get()

            try:
                receive_img_path = receive_data['data']
                decode_img_path = receive_img_path.decode(self.decode_encode)
                self.logger.info(f'Recive data: {decode_img_path}')

                result = self.function(decode_img_path)
                encode_result = result.encode(self.decode_encode)

                self.send_data(client_socket,receive_data, encode_result)

            except Exception as e:
                self.logger.info(f"Error: {e}. message: {decode_img_path} ")
                client_socket.close()
                self.client_connections.remove(client_socket)
                self.logger.info(f"Client close")