import numpy as np
import os
import cv2
from PIL import Image,ImageTk

import tkinter as tk
from tkinter import filedialog

import sys

# sys.path.append("..")
# sys.path.append("../LIIF")
# sys.path.append("../MAT")
# sys.path.append("../SAM")


# import LIIF
# import MAT
# import SAM                    

input_backImg_path = "./samples/input_img/background"
input_charImg_path = "./samples/input_img/charactor"
MAT_img_path = "./samples/MAT_img"
SAM_img_path = "./samples/SAM_img"
LIIF_img_path = "./samples/LIIF_img"


# image_char = 0
# image_background = 0
def handle_mouse_click(event):
    x = event.x
    y = event.y
    print("Mouse position - X:", x, "Y:", y)
    
def open_image1():
    # 파일 대화상자를 열어 이미지 파일 선택
    file_path = filedialog.askopenfilename()
    
    # OpenCV를 사용하여 이미지 로드
    image1  = cv2.imread(file_path)
    
    cv2.imwrite(input_backImg_path + "/image_bg.png", image1)

    # 이미지 크기 조정
    image1  = cv2.resize(image1 , (512, 512))
    
    # 이미지를 Tkinter에서 사용할 수 있는 형식으로 변환
    image1  = cv2.cvtColor(image1 , cv2.COLOR_BGR2RGB)
    image1  = Image.fromarray(image1)
    image1  = ImageTk.PhotoImage(image1)
        
    # 이미지를 화면에 표시
    widget0_0.configure(image=image1)
    widget0_0.image = image1 
    
    
def open_image2():
    # 파일 대화상자를 열어 이미지 파일 선택
    file_path = filedialog.askopenfilename()
    
    # OpenCV를 사용하여 이미지 로드
    image2  = cv2.imread(file_path)
    
    cv2.imwrite(input_charImg_path + "/image_ch.png", image2)

    # 이미지 크기 조정
    image2  = cv2.resize(image2 , (512, 512))
    
    # 이미지를 Tkinter에서 사용할 수 있는 형식으로 변환
    image2  = cv2.cvtColor(image2 , cv2.COLOR_BGR2RGB)
    image2  = Image.fromarray(image2)
    image2  = ImageTk.PhotoImage(image2)
        
    # 이미지를 화면에 표시
    widget1_0.configure(image=image2)
    widget1_0.image = image2 

# def inpaint_image():
#   MAT.module_inpainting.generate_images_ours(
#     dpath = SAM_img_path + "/origin",
#     mpath = SAM_img_path + "/mask",
#     outdir= MAT_img_path)

# def super_resolution():
#   LIIF.module_super_resolution.make_high_resolution(
#     img1_path = 'LIIF/images/image1.jpg',
#     img2_path = 'LIIF/images/image2.jpg',
#     output_path = 'LIIF/outputs/output1.jpg',
#     resolution_width = 800,
#     resolution_height = 800)


# Tkinter 창 생성
window = tk.Tk()

# 창 크기 설정
window.geometry("1800x1800")  

# 격자 레이아웃 설정
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=3)
window.grid_rowconfigure(0, weight=1)  # 추가된 부분
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)


widget0_0 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
widget0_0.grid(row=0, column=0, sticky="nsew")
# widget0_1 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
# widget0_1.grid(row=0, column=1, sticky="nsew")

widget1_0 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
widget1_0.grid(row=1, column=0, sticky="nsew")
# widget1_1 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
# widget1_1.grid(row=1, column=1, sticky="nsew")

widget2_0 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
widget2_0.grid(row=2, column=0, sticky="nsew")
# widget2_1 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
# widget2_1.grid(row=2, column=1, sticky="nsew")

widget_combined = tk.Frame(window, bg="white")
widget_combined.grid(row=0, column=1, sticky="nsew")
image_label = tk.Label(widget_combined, bg="white", fg="black", font=("Arial", 16))
image_label.pack()


image_label.bind("<Button-1>", handle_mouse_click)
# 버튼 생성
button1 = tk.Button(widget0_0, text="background img", command=open_image1)
button1.pack()
button2 = tk.Button(widget1_0, text="charactor img", command=open_image2)
button2.pack()


# 메인 이벤트 루프 시작
window.mainloop()
