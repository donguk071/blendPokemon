import numpy as np
import os
import cv2
from PIL import Image,ImageTk

import tkinter as tk
from tkinter import filedialog

import sys
import time
sys.path.append("..")

import LIIF
import MAT
import SAM                    

input_backImg_path = "./samples/input_img/background"
input_charImg_path = "./samples/input_img/charactor"
MAT_img_path = "./samples/MAT_img"
SAM_img_path = "./samples/SAM_img"
LIIF_img_path = "./samples/LIIF_img"


#SAM.generate_mask.generate_mask(image_path = "./samples/SAM_img/img_bg.png",sam_checkpoint = "../SAM/sam_vit_b_01ec64.pth",model_type= "vit_b" , device= "cpu")


isclicked = False
image_fin = None
image_label = None

def handle_mouse_click(event):
    global isclicked  # 전역 변수를 사용할 것임을 명시
    x = event.x
    y = event.y
    isclicked = True
    print("Mouse position - X:", x, "Y:", y)
  

def handle_mouse_move(event):
    global isclicked, image_fin, image_label
    if isclicked:
        x = event.x
        y = event.y
        print("Mouse position - X:", x, "Y:", y)
         
        image_path_b = "./samples/MAT_img/inpainted_bg.png"
        image_bg = cv2.imread(image_path_b ,cv2.IMREAD_COLOR)
        image_bg = cv2.resize(image_bg, dsize=(800,800))#리사이즈대신 superres
        
        image_path_f = "./samples/SAM_img/charactor/img_fg.png"
        image_fg = cv2.imread(image_path_f ,cv2.IMREAD_COLOR)
        image_fg = cv2.resize(image_fg, dsize=(800,800))#리사이즈대신 superres
        
        image_path_m = "./samples/SAM_img/mask/mask.png"
        image_m = cv2.imread(image_path_m, cv2.IMREAD_COLOR)
        image_m = cv2.resize(image_m, dsize=(800,800))#리사이즈대신 superres

        h, w = image_fg.shape[:2]
        
        center_h , center_w = int(h/2) , int(w/2)
        
        #todo
        new_x = x - center_w
        new_y = y - center_h
        
        # 이미지를 평행 이동하기 위해 변환 행렬 생성
        M = np.float32([[1, 0, new_x], [0, 1, new_y]])
        image_fg_translated = cv2.warpAffine(image_fg, M, (w, h))
        image_m_translated = cv2.warpAffine(image_m, M, (w, h))

        # 비트 연산자를 사용하여 이미지를 더해줌
        res = cv2.bitwise_and(image_bg, cv2.bitwise_not(image_m_translated))
        res = cv2.add(res, image_fg_translated)

        
        image_fin = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        image_fin = Image.fromarray(image_fin)
        image_fin = ImageTk.PhotoImage(image_fin)
        image_label.configure(image=image_fin)
        image_label.image = image_fin


        # time.sleep(1)


def handle_mouse_release(event):
    global isclicked  # 전역 변수를 사용할 것임을 명시
    isclicked = False


def open_image1():
    # 파일 대화상자를 열어 이미지 파일 선택
    file_path = filedialog.askopenfilename()
    
    # OpenCV를 사용하여 이미지 로드
    image1  = cv2.imread(file_path)
    
    # 이미지 크기 조정
    image1  = cv2.resize(image1 , (512, 512))
    
    cv2.imwrite(input_backImg_path + "/image_bg.png", image1)
    
    # 이미지를 Tkinter에서 사용할 수 있는 형식으로 변환\
    image1  = cv2.resize(image1 , (256, 256))
    image1  = cv2.cvtColor(image1 , cv2.COLOR_BGR2RGB)
    image1  = Image.fromarray(image1)
    image1  = ImageTk.PhotoImage(image1)
        
    # 이미지를 화면에 표시
    widget0_0.configure(image=image1)
    widget0_0.image = image1 
    
    #SAM.generate_mask.generate_mask(image_path = input_backImg_path + "/image_bg.png",sam_checkpoint = "../SAM/sam_vit_b_01ec64.pth",model_type= "vit_b" , device= "cpu")
    SAM.generate_mask.generate_mask(image_path = input_backImg_path + "/image_bg.png", device= "cpu")

    image_path = SAM_img_path + "/img_bg.png"
    image = Image.open(image_path)
    image_tk = ImageTk.PhotoImage(image)
    widget2_0.configure(image=image_tk)
    widget2_0.image = image_tk 
    

def open_image2():
    # 파일 대화상자를 열어 이미지 파일 선택
    file_path = filedialog.askopenfilename()
    
    # OpenCV를 사용하여 이미지 로드
    image2  = cv2.imread(file_path)
    
    # 이미지 크기 조정
    image2  = cv2.resize(image2 , (512, 512))
    
    cv2.imwrite(input_charImg_path + "/image_ch.png", image2)
    
    # 이미지를 Tkinter에서 사용할 수 있는 형식으로 변환
    image2  = cv2.resize(image2 , (256, 256))
    image2  = cv2.cvtColor(image2 , cv2.COLOR_BGR2RGB)
    image2  = Image.fromarray(image2)
    image2  = ImageTk.PhotoImage(image2)
        
    # 이미지를 화면에 표시
    widget1_0.configure(image=image2)
    widget1_0.image = image2 
    
    #SAM.generate_mask.generate_mask(image_path = input_charImg_path + "/image_ch.png" ,sam_checkpoint = "../SAM/sam_vit_b_01ec64.pth",model_type= "vit_b" , device= "cpu")
    SAM.generate_mask.generate_mask(image_path = input_charImg_path + "/image_ch.png" , device= "cpu")

    image_path = SAM_img_path + "/charactor/img_fg.png"
    image = Image.open(image_path)
    image_tk = ImageTk.PhotoImage(image)
    widget2_0.configure(image=image_tk)
    widget2_0.image = image_tk 

def inpaint_image() : 
    # MAT.module_inpainting.generate_images_ours(
    #     dpath = SAM_img_path + "/origin", #sam 저장경로로 수정하던지 sam 결과가 저기 저장되도록
    #     mpath = SAM_img_path + "/mask",
    #     outdir= MAT_img_path)
    global image_label
    image_path = "./samples/SAM_img/img_bg.png"
    image = Image.open(image_path)
    image_tk = ImageTk.PhotoImage(image)
    image_label = tk.Label(col1_frame, image=image_tk)
    image_label.bind("<Motion>", handle_mouse_move)
    image_label.bind("<Button-1>", handle_mouse_click)
    image_label.bind("<ButtonRelease-1>", handle_mouse_release)
    image_label.pack()

    image_path = MAT_img_path + "/inpainted_bg.png"
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image_tk = ImageTk.PhotoImage(image)
    
    widget2_0.configure(image=image_tk)
    widget2_0.image = image_tk 

def show_result():

    # image_path = "./samples/SAM_img/img_bg.png"
    # image = Image.open(image_path)
    # image_tk = ImageTk.PhotoImage(image)
    # image_label = tk.Label(col1_frame, image=image_tk)
    # image_label.bind("<Motion>", handle_mouse_move)
    # image_label.bind("<Button-1>", handle_mouse_click)
    # image_label.bind("<ButtonRelease-1>", handle_mouse_release)
    # image_label.pack()
    global image_label

    image_path = MAT_img_path + "/inpainted_bg.png"
    image = Image.open(image_path)
    image_tk = ImageTk.PhotoImage(image)
    
    image_label.configure(image=image_tk)
    image_label.image = image_tk 
    
    

# Tkinter 창 생성
window = tk.Tk()

# 창 크기 설정
# window.geometry("1800x1600")  

# 격자 레이아웃 설정
window.grid_columnconfigure(0, minsize=300)  # 열 0의 최소 크기를 200으로 지정
window.grid_columnconfigure(1, minsize=900)  # 열 1의 최소 크기를 600으로 지정
window.grid_rowconfigure(0, minsize=300)     # 행 0의 최소 크기를 200으로 지정
window.grid_rowconfigure(1, minsize=300)     # 행 1의 최소 크기를 200으로 지정
window.grid_rowconfigure(2, minsize=300)     # 행 2의 최소 크기를 200으로 지정


col1_frame = tk.Frame(window)
col1_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
# col1_frame.bind("<Button-1>", handle_mouse_click)

#마우스 이벤트 받도록 바인딩
widget0_0 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
widget0_0.grid(row=0, column=0, sticky="nsew")
widget0_0.bind("<Button-1>", handle_mouse_click)

widget1_0 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
widget1_0.grid(row=1, column=0, sticky="nsew")
widget1_0.bind("<Button-1>", handle_mouse_click)

widget2_0 = tk.Label(window, bg="white", fg="black", font=("Arial", 16))
widget2_0.grid(row=2, column=0, sticky="nsew")

# 버튼 생성
button1 = tk.Button(widget0_0, text="background img", command=open_image1)
button1.pack()
button2 = tk.Button(widget1_0, text="charactor img", command=open_image2)
button2.pack()
button2 = tk.Button(widget2_0, text="inpaint background img", command=inpaint_image)
button2.pack()
button3 = tk.Button(col1_frame, text="super resolution and show result", command=show_result)
button3.pack()

# 메인 이벤트 루프 시작
window.mainloop()
