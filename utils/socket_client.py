import socket
import struct

def main():
    host = '127.0.0.1'  # 服务器的IP地址
    port = 8887  # 服务器的端口号

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    try:
        # 准备要发送的数据
        data_to_send = "D:/Users/Desktop/tmp/1stitching11.png*1"
        send_data(client_socket, data_to_send)

        # 接收服务器的响应
        received_data = receive_data(client_socket)
        print("Received from server:", received_data['echo'])
    except Exception as e:
        print("Error:", e)
    finally:
        client_socket.close()

def send_data(client_socket, data):
    # 准备数据结构
    header = {
        'head': 0,
        'num': 0,
        'cmd': 0,
        'status': 0,
        'encrypt': 0,
        'len': len(data)
    }
    send_struct = {
        'header': header,
        'data': data,
        'echo': ''
    }

    # 打包数据
    header_format = f"{len(header.keys())}I"
    pack_head = struct.pack(header_format, *header.values())
    data_format = f"{header['len']}s"
    pack_data = struct.pack(data_format, data.encode())

    # 发送数据
    client_socket.sendall(pack_head)
    client_socket.sendall(pack_data)

def receive_data(client_socket):
    # 接收头部
    header_format = "6I"
    header_size = struct.calcsize(header_format)
    header_data = client_socket.recv(header_size)
    header_unpack = struct.unpack(header_format, header_data)

    # 接收数据
    data_size = header_unpack[-1]
    data_format = f"{data_size}s"
    data = client_socket.recv(data_size)
    data_unpack = struct.unpack(data_format, data)

    # 构建接收到的数据结构
    received_data = {
        'header': {
            'head': header_unpack[0],
            'num': header_unpack[1],
            'cmd': header_unpack[2],
            'status': header_unpack[3],
            'encrypt': header_unpack[4],
            'len': header_unpack[5]
        },
        'data': data_unpack[0].decode(),
        'echo': ''
    }

    return received_data

if __name__ == "__main__":
    main()
