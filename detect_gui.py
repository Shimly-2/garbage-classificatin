import argparse
import time
from pathlib import Path
from tkinter import image_names

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import time,sys
from tkinter.font import Font
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets, QtCore, QtGui
import json
import cv2
import os
import numpy as np
from ctypes import *
import datetime
from PIL import Image

global save_path
save_path=''

def detect(opt,save_img=False):
    global save_path
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir    

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string
                
                labeli=['','','','','','','']
                numm=0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        labeli[numm]=label
                        numm=numm+1
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    timess=time.time() - t0
    print(f'Done. ({time.time() - t0:.3f}s)')
    return labeli,numm,timess

class WinForm(QGraphicsView,QMainWindow):
	# 视频播放列表
	bClose = False
	fn_list = ["C:/Users/17628/Desktop/ui/test video/慢速版.mp4","C:/Users/17628/Desktop/ui/test video/正常版.mp4"]
	play_index = 0
	def __init__(self, parent=None):
		super(WinForm, self).__init__(parent)
		print("初始化yolov5...")
		self.setWindowTitle('智能垃圾分类')  # 设置窗口标题
		# self.setWindowTitle("参数跟踪")
		self.setStyleSheet("background-color:#EDEDF5")
		self.setGeometry(0, 0, 640, 480)  # 窗口整体窗口位置大小

		self.label_13 = QLabel(self)
		self.label_13.setText("显示图片")
		self.label_13.setFixedSize(370, 277)
		self.label_13.move(5, 70)  #10  485

		self.label_14 = QLabel(self)
		self.label_14.setText("0.000")
		self.label_14.setFixedSize(50, 20)
		self.label_14.move(370+35+60,400+45)  #10  485
		# self.label_14.setFont(font)
		# self.label_14 = QLabel(self)
		# self.label_14.setText("显示图片")
		# self.label_14.setFixedSize(640/2, 480/2)
		# self.label_14.move(5, 480/2+10)  #485

		# 设置字体类
		font = QtGui.QFont()
		font.setFamily("Consolas")
		font.setPointSize(10)
		font.setBold(False)
		font.setItalic(False)
		font.setWeight(3)

		font2 = QtGui.QFont()
		font2.setFamily("Consolas")
		font2.setPointSize(24)
		font2.setBold(False)
		font2.setItalic(False)
		font2.setWeight(3)
		self.label_14.setFont(font)

		# 设置滚动条布局
		self.topFiller2 = QWidget()
		self.topFiller2.setMinimumSize(250, 2000)  # 设置滚动条的尺寸
		# self.label18.setFont(font)

		global labelnum
		labelnum= []
		for i in range(120):
			num = i
			var = 'Lh' + str(num)
			labelnum.append(var)

		# 动态生成边线行数label，同时设定对应的布局
		for filename in range(12):
			self.label20 = QLabel(self.topFiller2)
			self.label20.setObjectName(labelnum[filename])
			self.label20.setText(" ")
			self.label20.move(10,filename*30+20)
			self.label20.setFont(font)

		# 设置左中右边线参数列表，初始化
		global labelLeft,labelRight,labelMid
		labelLeft = []
		for i in range(120):
			num = i
			var = 'L' + str(num)
			labelLeft.append(var)
		labelMid = []
		for i in range(120):
			num = i
			var = 'M' + str(num)
			labelMid.append(var)
		labelRight = []
		for i in range(120):
			num = i
			var = 'R' + str(num)
			labelRight.append(var)

		# 动态生成左中右边线列表的label，同时设定对应的布局
		for i in range(12):
			self.label100=QLabel(self.topFiller2)
			self.label100.setObjectName(labelLeft[i])
			self.label100.setText("      ")
			self.label100.move(50,20+i*30)
			self.label100.setFont(font)
			self.label100.setStyleSheet("color:#2A78D6")

			self.label200=QLabel(self.topFiller2)
			self.label200.setObjectName(labelMid[i])
			self.label200.setText("    ")
			self.label200.move(125,20+i*30)
			self.label200.setFont(font)
			self.label200.setStyleSheet("color:#782AD6")

			self.label300=QLabel(self.topFiller2)
			self.label300.setObjectName(labelRight[i])
			self.label300.setText("    ")
			self.label300.move(180,20+i*30)
			self.label300.setFont(font)
			self.label300.setStyleSheet("color:#45C937")

        # 边线列表头
		self.label18=QLabel(self.topFiller2)
		self.label18.setText("序号   垃圾类别    数量  分类成功与否\n--------------------------------------------------------")
		self.label18.setStyleSheet("color:#F46F0C")
		self.label18.resize(300,50)
		self.label18.move(5,0)

		# 将布局放置在滚动条中
		self.scrol2 = QScrollArea(self)
		self.scrol2.setWidget(self.topFiller2)
		self.scrol2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.scrol2.resize(250,400)
		self.scrol2.move(380,30)

		self.btn = QPushButton(self)
		self.btn.setText("START")
		self.btn.move(10, 30)
		self.btn.setStyleSheet("QPushButton{background-color: rgb(200,200,230);border:2px groove gray;border-radius:10px;padding:2px 4px;border-style: outset;}"
											"QPushButton:hover{background-color:rgb(229, 241, 251); color: black;}"
											"QPushButton:pressed{background-color:rgb(204, 228, 247);border-style: inset;}")
		self.btn.clicked.connect(self.start_event)
		self.btn.resize(120,30)
		self.btn.setFont(font)

		# self.btn2 = QPushButton(self)
		# self.btn2.setText("打开视频")
		# self.btn2.move(135, 30)
		# self.btn2.setStyleSheet("QPushButton{background-color: rgb(200,200,230);border:2px groove gray;border-radius:10px;padding:2px 4px;border-style: outset;}"
		# 									"QPushButton:hover{background-color:rgb(229, 241, 251); color: black;}"
		# 									"QPushButton:pressed{background-color:rgb(204, 228, 247);border-style: inset;}")
		# self.btn2.clicked.connect(self.manual_choose)
		# self.btn2.resize(120,40)
		# self.btn2.setFont(font)

		self.screen = QDesktopWidget().screenGeometry()
		self.resize(self.screen.width(), self.screen.height())

		# 设置菜单栏
		bar = QMenuBar(self)
		bar.setFixedSize(self.screen.width(),24)
		# 设置选项二
		play_menu = bar.addMenu("视频调试")
		# 设置按钮一
		play_video = QAction("打开视频文件",self)
		play_video.setShortcut("Ctrl+1")
		play_video.triggered.connect(self.manual_choose)
		play_menu.addAction(play_video)
		# 设置按钮二
		play_pictures = QAction("自动播放视频",self)
		play_pictures.setShortcut("Ctrl+2")
		play_pictures.triggered.connect(self.auto_choose)
		play_menu.addAction(play_pictures)
		# 设置停止按钮
		play_stop = QAction("停止",self)
		play_stop.setShortcut("Ctrl+3")
		play_stop.triggered.connect(self.video_stop)
		play_menu.addAction(play_stop)
		# 设置退出按钮
		exit_menu = bar.addMenu("图片调试")
		exit_option = QAction("打开图片文件",self)
		exit_option.setShortcut("Ctrl+Q")
		exit_option.triggered.connect(self.openimage)
		exit_menu.addAction(exit_option)
		# 设置菜单栏的位置
		bar.move(0,0)

		# 设置滑动条
		self.s1 = QSlider(Qt.Horizontal,self)
		self.s1.setToolTip("滑动条")
		self.s1.setMinimum(0)  # 设置最大值
		self.s1.setMaximum(50)  # 设置最小值
		self.s1.setSingleStep(1)  # 设置间隔
		self.s1.setValue(0)  # 设置当前值
		self.s1.sliderMoved.connect(self.start_drag)
		self.s1.sliderReleased.connect(self.drag_action)
		self.s1.setFixedSize(360, 20)
		self.moving_flag = 0
		self.stop_flag = 0  # 如果当前为播放值为0,如果当前为暂停值为1
		self.s1.move(10,400+40)
		# 设置两个标签分别是当前时间和结束时间
		self.label_start = QLabel("00:00",self)
		self.label_start.move(10,400+60)
		self.label_start.setFont(font)
		self.label_end = QLabel("00:00",self)
		self.label_end.setFont(font)
		self.label_end.move(370-30,400+60)
		# 设置暂停播放和下一个按钮
		self.stop_button = QPushButton(self)
		self.stop_button.setIcon(QIcon('C:/Users/17628/Desktop/ui/icon/开始.png'))
		self.stop_button.setIconSize(QSize(20,20))
		self.stop_button.clicked.connect(self.stop_action)
		self.stop_button.move(370+15,400+40)
		self.next_button = QPushButton(self)
		self.next_button.setIcon(QIcon('C:/Users/17628/Desktop/ui/icon/下一个.png'))
		self.next_button.setIconSize(QSize(20,20))
		self.next_button.clicked.connect(self.next_action)
		self.next_button.move(370+15+35,400+40)
		self.setGeometry(0, 0, 640, 480)  # 窗口整体窗口位置大小
		self.show()

	# 定义将opencv图像转PyQt图像的函数
	def cvImgtoQtImg(self,cvImg):
		QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
		QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0],QtGui.QImage.Format_RGB32)
		return QtImg

	# 播放视频进行参数跟踪，目前只有OTSU的二值化图像
	def playVideoFile(self,fn):
		self.cap = cv2.VideoCapture(fn)
		frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.settingSlider(frames)
		fps = 24
		self.loop_flag = 0
		if not self.cap.isOpened():
			print("Cannot open Video File")
			exit()
		while not self.bClose:
			ret, frame = self.cap.read()  # 逐帧读取影片
			if not ret:
				if frame is None:
					print("The video has end.")
				else:
					print("Read video error!")
				break
			if self.moving_flag==0:
				self.label_start.setText(self.int2time(self.loop_flag))
				self.s1.setValue(int(self.loop_flag/24))#设置当前值
			self.loop_flag += 1

			# 处理得到二值化图像
			global side,md
			mat_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			gray = cv2.cvtColor(mat_img, cv2.COLOR_RGB2GRAY)
			frame = np.array(gray, dtype=np.uint8)
			array = frame.astype(c_uint8)
			a=c_uint8*(120*188+1)
			t=a()
			mylib.Get_01_Value.restype=POINTER(c_uint8)
			t1=mylib.Get_01_Value(array.ctypes.data_as(POINTER(c_uint8)),byref(t),4)
			for i in range(120):
				m=0
				for j in range(188*i,(i+1)*188):
					aa[i][m]=t1[j]
					if(aa[i][m]==1):
						aa[i][m]=0
					m=m+1
			# 得到边线数组
			m=0
			for i in range(56,120):
				d=0
				for j in range(188):
					md[m][d]=aa[i][j]
					d=d+1
				m=m+1
			frame = np.array(md, dtype=np.uint8)
			array = frame.astype(c_uint8)
			a=c_int8*64*3
			t=a()
			mylib.Get_line_LMR.restype=POINTER(c_uint8)
			t1=mylib.Get_line_LMR(array.ctypes.data_as(POINTER(c_uint8)),byref(t))
			for i in range(3):
				m=0
				for j in range(64*i,(i+1)*64):
					side[m][i]=t1[j]
					m=m+1
			# 创建一个新的cv2格式图像存储二值化图像
			img_binary=create_image_singel(aa)
			# 转化到label支持的图像格式
			QtImg = self.cvImgtoQtImg(img_binary)
			# 显示二值化图像
			self.label_13.setPixmap(QtGui.QPixmap.fromImage(QtImg).scaled(self.label_13.size()))
			self.label_13.show()  # 刷新界面

			# 创建一个新的cv2格式图像存储边线图像
			img=create_image(side)
			# 转化到label支持的图像格式
			QtImg = self.cvImgtoQtImg(img)
			# 显示二值化图像
			self.label_14.setPixmap(QtGui.QPixmap.fromImage(QtImg).scaled(self.label_14.size()))
			self.label_14.show()  # 刷新界面

			# 用于参数追踪
			a=c_float*2
			t=a()
			mylib.control.restype=POINTER(c_float)
			omega=c_float(1.0)
			cha=side[ROAD_MAIN_ROW][1]-93
			# 输出舵机转角以及车身倾角信息
			t1=mylib.control(omega,cha,byref(t),0)
			if(cha<0):
				t1[0]=t1[0]-90
				t1[1]=t1[1]-90
			# 进行可视化
			self.label_7.setText(str(round(t1[0],2)))
			self.rota_top.car_top_item.setRotation(t1[0])  # 自身改变旋转度
			self.label_10.setText(str(round(t1[1],2)))
			self.rota.car_back_item.setRotation(t1[1])  # 自身改变旋转度

			# 进行边线数组以及基本参数的可视化
			global labelLeft,labelRight,labelMid,label_canshu_value
			for i in range(len(label_canshu_value)):
				aaa=self.findChild(QLabel,label_canshu_value[i])
				aaa.setText(str(t1[0]))
			# 进行边线数组以及基本参数的可视化
			for i in range(64):
				aaa=self.findChild(QLabel,labelLeft[i])
				aaa.setText(str(side[i][0]))
				aaa=self.findChild(QLabel,labelMid[i])
				aaa.setText(str(side[i][1]))
				aaa=self.findChild(QLabel,labelRight[i])
				aaa.setText(str(side[i][2]))

			while self.stop_flag == 1:  # 暂停的动作
				cv2.waitKey(int(1000/fps))  # 休眠一会，因为每秒播放24张图片，相当于放完一张图片后等待41ms
			cv2.waitKey(int(1000/fps))  # 休眠一会，因为每秒播放24张图片，相当于放完一张图片后等待41ms
        # 释放
		self.cap.release()

	# 暂停触发
	def stop_action(self):
		if self.stop_flag == 0:
			self.stop_flag = 1
			self.stop_button.setIcon(QIcon('C:/Users/17628/Desktop/ui/icon/暂停.png'))
		else:
			self.stop_flag = 0
			self.stop_button.setIcon(QIcon('C:/Users/17628/Desktop/ui/icon/开始.png'))

	# 下一个触发
	def next_action(self):
		self.bClose = True
		self.play_index = (self.play_index+1)%3
		self.bClose = False
		self.playVideoFile(self.fn_list[self.play_index])

    # 滑动条触发
	def start_drag(self):
		self.moving_flag = 1

	# 拖动行为
	def drag_action(self):
		self.moving_flag = 0
		print('当前进度为%d，被拉动到的进度为%d'%(self.s1.value(), int(self.loop_flag/24)))
		if self.s1.value()!=int(self.loop_flag/24):
			print('当前进度为:'+str(self.s1.value()))
			self.loop_flag = self.s1.value()*24
			self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_flag)

	# 暂停行为
	def video_stop(self):
		self.bClose = True

	# 手动选择视频播放
	def manual_choose(self):
		fn,_ = QFileDialog.getOpenFileName(self,'Open file','C:/Users/17628/Desktop/ui/test video',"Video files (*.mp4 *.avi)")
		self.playVideoFile(fn)

	# 自动选择视频播放，如何自动视频播放？
	def auto_choose(self):
		self.playVideoFile(self.fn_list[self.play_index])

	# 设置进度条
	def settingSlider(self,maxvalue):
		self.s1.setMaximum(int(maxvalue/24))
		self.label_end.setText(self.int2time(maxvalue))

	# 视频播放中的时间函数
	def int2time(self,num):
        # 每秒刷新24帧
		num = int(num/24)
		minute = int(num/60)
		second = num - 60*minute
		if minute < 10:
			str_minute = '0'+str(minute)
		else:
			tr_minute = str(minute)
		if second < 10:
			str_second = '0'+str(second)
		else:
			str_second = str(second)
		return str_minute+":"+str_second

	# 暂停行为
	def exit_stop(self):
		self.stop_flag = 0
		self.bClose = True
		self.close()

    # 打开图片，有bug，未调，不需要
	def openimage(self):
		global imgName,save_path
		imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.bmp;;*.png;;All Files(*)")
		# 将图片转化为label支持的图像格式
		jpg = QtGui.QPixmap(imgName).scaled(self.label_13.width(), self.label_13.height())
		# 显示在label上
		self.label_13.setPixmap(jpg)

	def start_event(self):
		global imgName,save_path
		img=imgName
		parser = argparse.ArgumentParser()
		parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
		parser.add_argument('--source', type=str, default=img, help='source')  # file/folder, 0 for webcam   'data/images'
		parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
		parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
		parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
		parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
		parser.add_argument('--view-img', action='store_true', help='display results')
		parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
		parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
		parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
		parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
		parser.add_argument('--augment', action='store_true', help='augmented inference')
		parser.add_argument('--update', action='store_true', help='update all models')
		parser.add_argument('--project', default='runs/detect', help='save results to project/name')
		parser.add_argument('--name', default='exp', help='save results to project/name')
		parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
		opt = parser.parse_args()
		print(opt)
		with torch.no_grad():
			if opt.update:  # update all models (to fix SourceChangeWarning)
				for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
					labeli,numm,timess=detect(opt)
					strip_optimizer(opt.weights)
			else:
				labeli,numm,timess=detect(opt)
        # 将图片转化为label支持的图像格式
		jpg = QtGui.QPixmap(save_path).scaled(self.label_13.width(), self.label_13.height())
		# 显示在label上
		self.label_13.setPixmap(jpg)
		print(labeli)
		print(timess)
		i=1
        # 动态生成左中右边线列表的label，同时设定对应的布局
		for m in range(0,numm):
			aaa=self.findChild(QLabel,labelnum[i])
			aaa.setText(str(i))
			aaa=self.findChild(QLabel,labelLeft[i])
			aaa.setText(labeli[m])
			aaa=self.findChild(QLabel,labelMid[i])
			aaa.setText('1')
			aaa=self.findChild(QLabel,labelRight[i])
			aaa.setText('成功')
			i=i+1

		self.label_14.setText(str(round(timess,3)))





		# imgptr = self.label.pixmap().toImage()
		# ptr = imgptr.constBits()
		# ptr.setsize(imgptr.byteCount())
		# mat = np.array(ptr).reshape(imgptr.height(), imgptr.width(), 4)
		# mat_img = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
		# gray = cv2.cvtColor(mat_img, cv2.COLOR_RGB2GRAY)
		# ret, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

		# frame = np.array(binary, dtype=np.uint8)
		# array = frame.astype(c_uint8)
		# a=c_uint8*120*2
		# t=a()
		# # mylib.ImageGetSide.restype=c_char_p
		# # t1=mylib.ImageGetSide(array.ctypes.data_as(POINTER(c_uint8)),byref(t))
		# aa=[]
		# m=0
		# for i in range(120):
		# 	aa.append([])
		# 	for j in range(2):
		# 		aa[i].append(t1[m])
		# 		m=m+1
		# binary = Image.fromarray(np.uint8(binary))
		# binary = binary.toqpixmap() #QPixmap
		# self.label_13.setPixmap(binary)

# global img
# img='data/images/bus.jpg'
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
# parser.add_argument('--source', type=str, default=img, help='source')  # file/folder, 0 for webcam   'data/images'
# parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
# parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--view-img', action='store_true', help='display results')
# parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
# parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
# parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# parser.add_argument('--update', action='store_true', help='update all models')
# parser.add_argument('--project', default='runs/detect', help='save results to project/name')
# parser.add_argument('--name', default='exp', help='save results to project/name')
# parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
# opt = parser.parse_args()
# print(opt)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    newWin = WinForm()
    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()

    sys.exit(app.exec_())