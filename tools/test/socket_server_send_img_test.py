import socket
import cv2
import base64


def data_convert(path):
    img = cv2.imread(path)
    png_bytes = cv2.imencode('.png', img)[1].tobytes()
    # 使用Base64编码将字节对象转换为字符串
    base64_string = base64.b64encode(png_bytes)
    decode = base64_string.decode('utf-8')
    encode = decode.encode('utf-8')
    return encode,png_bytes

# 创建一个IPv4 TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定服务器地址和端口
host = '127.0.0.1'  # 服务器IP地址
port = 8886  # 服务器端口
server_socket.bind((host, port))

# 监听连接，参数是最大连接数
server_socket.listen(5)
print(f"服务器正在监听 {host}:{port}")

while True:
    # 等待客户端连接
    client_socket, client_address = server_socket.accept()
    print(f"接收到来自 {client_address} 的连接")

    # 接收客户端消息
    data = client_socket.recv(1024)
    if not data:
        break  # 如果没有数据，退出循环

    print(f"接收到消息：{data.decode('utf-8')}")

    # 发送响应给客户端10
    img_path = "F:/adam/i1000/adk/adk_src/0_1_26_area.jpg"
    response,png_bytes = data_convert(img_path)
    # encode_data,png_bytes = response.encode('utf-8')
    client_socket.send(png_bytes)

    # 关闭客户端连接
    client_socket.close()

# 关闭服务器socket
server_socket.close()
