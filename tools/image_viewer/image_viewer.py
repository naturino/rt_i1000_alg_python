import cv2
import copy
import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk

class TkinterWindows:
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)

    def add_check_button(self, text, function, row, column, padx=10, pady=10):
        checkbox_var = tk.IntVar()
        checkbox = tk.Checkbutton(self.root, text=text, variable=checkbox_var, command=function)
        checkbox.grid(row=row, column=column, padx=padx, pady=pady)
        return checkbox

    def add_button(self, text, function, row, column, padx=10, pady=10, sticky="e"):
        button = tk.Button(self.root, text=text, command=function)
        button.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
        return button

    def add_scale(self, text, function, row, column, padx=10, pady=10, from_=100, to=1000, orient="horizontal",
                  length=1000, default_value=0,image = np.array([])):
        default_value = tk.IntVar(value=default_value)
        scale = tk.Scale(self.root, label=text, from_=from_, to=to, orient=orient, length=length,
                      variable=default_value)
        scale.grid(row=row, column=column, padx=padx, pady=pady)
        scale.config(image, command=function)
        return scale

    def add_frame(self, row, column, columnspan, padx=10, pady=10):
        frame = tk.Frame(self.root)
        frame.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady)
        return frame

    def add_canvas(self, row, column, columnspan, padx=10, pady=10, width=512, height=512, bg="white"):
        frame = self.add_frame(row, column, columnspan, padx, pady)
        canvas = tk.Canvas(frame, bg=bg, width=width, height=height)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=tk.Frame(canvas), anchor="nw")

        return canvas

class ImageEhance:
    def __init__(self):
        pass

    def resize_image(self, src, val):
        scale_value = int(val)
        dst = src
        if src.size > 0:
            dst = cv2.resize(src, (scale_value, scale_value))

        return dst

    def gaussian_image(self, src, val):
        value = int(val)
        dst = src
        if src.size > 0:
            for i in range(value):
                dst = cv2.GaussianBlur(dst, (3, 3), 1)

        return dst

    def sharpen_image(self, src, val):
        value = int(val)
        dst = src
        if src.size > 0:
            op = np.array([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ], dtype=np.float32)
            for i in range(value):
                dst = cv2.filter2D(dst, cv2.CV_32F, op)
                dst = cv2.convertScaleAbs(dst)
            # blur_img = cv2.GaussianBlur(dst, (0, 0), 25)
            # dst = cv2.addWeighted(dst, 1.5, blur_img, -0.5, 0)
            #
            # blur_laplace = cv2.Laplacian(dst, -1)
            # dst = cv2.addWeighted(dst, 1, blur_laplace, -0.5, 0)

        return dst

class ImageViewerApp:
    def __init__(self, root, title):
        self.root = root
        self.viewer_windows = TkinterWindows(root, title)
        self.image_ehance = ImageEhance()

        self.src = np.array([])
        self.dst = np.array([])

        self.init_value = 0
        self.sharpen_value = self.init_value
        self.gaussian_value = self.init_value

        self.frame_size = 512

        self.init_windows()

    def init_windows(self):
        self.sharpen_scale = self.viewer_windows.add_scale("锐化", self.sharpen_scale_function, row=0, column=0,
                                                           from_=0, to=10, length=self.frame_size,
                                                           default_value=self.init_value)
        self.gaussian_scale = self.viewer_windows.add_scale("模糊", self.gaussian_scale_function, row=0, column=1,
                                                            from_=0, to=20, length=self.frame_size,
                                                            default_value=self.init_value)
        self.import_button = self.viewer_windows.add_button("导入图片", self.import_button_function, row=3, column=0,
                                                            sticky='s')
        self.reload_button = self.viewer_windows.add_button("重新加载", self.reload_button_function, row=3, column=1,
                                                            sticky='s')

        self.src_frame = self.viewer_windows.add_canvas(row=2, column=0, columnspan=1, width=self.frame_size,
                                                        height=self.frame_size)
        self.dst_frame = self.viewer_windows.add_canvas(row=2, column=1, columnspan=1, width=self.frame_size,
                                                        height=self.frame_size)

    def run(self):
        gaussian_img = copy.deepcopy(self.src)
        if self.gaussian_value != self.init_value:
            gaussian_img = self.image_ehance.gaussian_image(gaussian_img, self.gaussian_value)

        sharpen_img = copy.deepcopy(gaussian_img)
        if self.sharpen_value != self.init_value:
            sharpen_img = self.image_ehance.sharpen_image(sharpen_img, self.sharpen_value)

        self.dst = sharpen_img
        self.dst_label = self.display_image(self.dst, self.dst_frame)

    def sharpen_scale_function(self, val):
        self.sharpen_value = val
        self.run()

    def sharpen_check_button_function(self,is_open):
        if is_open:
            sharpen_img = self.image_ehance.sharpen_image(self.dst, self.sharpen_value)

            self.dst = sharpen_img
        self.dst_label = self.display_image(self.dst, self.dst_frame)

    def gaussian_scale_function(self, val):
        self.gaussian_value = val
        self.run()

    def reload_button_function(self):
        self.dst = copy.deepcopy(self.src)
        self.sharpen_value = self.init_value
        self.gaussian_value = self.init_value

        self.dst_label = self.display_image(self.dst, self.dst_frame)

    def display_image(self, image, frame):
        if frame:
            frame.delete("image")  # 删除之前的图像

            # 获取Canvas的宽度和高度
            canvas_width = frame.winfo_width()
            canvas_height = frame.winfo_height()

            # 计算图像的位置，使其位于窗口中心
            if image.shape:
                img_pil = Image.fromarray(image)
                img_width, img_height = img_pil.size
                img_tk = ImageTk.PhotoImage(image=img_pil)

                x = (canvas_width - img_width) // 2
                y = (canvas_height - img_height) // 2

                # 在Canvas上显示图像
                frame.create_image(x, y, anchor=tk.NW, image=img_tk, tags="image")
                return img_tk

    def import_button_function(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (self.frame_size, self.frame_size))
            self.src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sharpen_img = copy.deepcopy(self.src)
            self.gaussian_img = copy.deepcopy(self.src)
            self.dst = copy.deepcopy(self.src)
            self.src_label = self.display_image(self.src, self.src_frame)
            self.dst_label = self.display_image(self.dst, self.dst_frame)


if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    title = "Image Viewer"
    app = ImageViewerApp(root, title)
    root.mainloop()
