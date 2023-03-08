# pyuic5 -o gui.py untitled.ui
import cv2, sys, yaml, os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from gui import Ui_Dialog
from PIL import Image
from yolo import yolov7
from PyQt5.QtGui import *
import torch
from scripts.test_pca9685 import disposal
import itertools
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt#设置背景色
import time


def resize_img(img, img_size=560, value=[255, 255, 255], inter=cv2.INTER_AREA):
    old_shape = img.shape[:2]
    ratio = img_size / max(old_shape)
    new_shape = [int(s * ratio) for s in old_shape[:2]]
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=inter)
    delta_h, delta_w = img_size - new_shape[0], img_size - new_shape[1]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), borderType=cv2.BORDER_CONSTANT,
                             value=value)
    return img


def all_np(arr):
    #拼接数组函数
    List = list(itertools.chain.from_iterable(arr))
    arr = np.array(List)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


label_list = ['battery', 'can', 'd','carrot', 'ceramic', 'cobblestone','potato', 'ternip', 'waterbottle']
def tran(label):
    cat = 0
    if label == 'battery':
        cat = 1
    elif label == 'can' or label == 'waterbottle':
        cat = 2
    elif label == 'carrot' or label == 'potato' or label == 'ternip':
        cat = 3
    elif label == 'ceramic' or label == 'cobblestone':
        cat = 4
    return cat


class MyForm(QDialog):
    def __init__(self, title):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        self.save_path = 'result'
        self.save_id = 0
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.now = None
        self.model = None
        self.video_count = None
        self._timer = None
        self._timer_video = None
        self.ui.textBrowser.setFontPointSize(12)
        
        self.ui.label.setText(title)
        # self.ui.pushButton_Model.clicked.connect(self.select_model)
        self.ui.pushButton_Img.clicked.connect(self.select_image_file)
        # self.ui.pushButton_ImgFolder.clicked.connect(self.select_folder_file)
        # self.ui.pushButton_Video.clicked.connect(self.select_video_file)
        self.ui.pushButton_Camera.clicked.connect(self.select_camear)
        self.ui.pushButton_BegDet.clicked.connect(self.begin_detect)
        # self.ui.pushButton_Exit.clicked.connect(self._exit)
        self.ui.pushButton_SavePath.clicked.connect(self.show_video_only)
        self.ui.pushButton_StopDet.clicked.connect(self.stop_detect)

        fileName = 'weight/best.pt'
        self.ui.textBrowser.append(f'load model form {fileName}.')
        # read cfg
        with open('model.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        # update model_path
        cfg['model_path'] = fileName
        # init yolov7 model
        self.model = yolov7(**cfg)
        self.ui.textBrowser.append(f'load model success.')

        self.show_camera_flag = 0
        self.show_video_only_flag = 0 

        # self._sensor = QTimer(self)
        # self._sensor.timeout.connect(self.signel_det)
        # self._sensor.start(20)
        # self.sensor_cont = 0
        self.select_camear()
        self.obj_cont = 1

        self.label_all = [[],[],[],[],[]]
        self.label_count = 0

        self.multily_drop = 0
        self.drop_1 = 0
        self.drop_2 = 0
        self.drop_3 = 0

        self.get_ready = 0

        self.show()
    
    def signel_det(self):
        if self.sensor_cont == 300:
            self._sensor.stop()
            self.select_camear()
            self.sensor_cont = 0
        else:
            self.sensor_cont += 1
        print(self.sensor_cont)

    def read_and_show_image_from_path(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        resize_image = cv2.cvtColor(resize_img(image), cv2.COLOR_RGB2BGR)
        self.ui.label_ori.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(resize_image.data, resize_image.shape[1], resize_image.shape[0], QtGui.QImage.Format_RGB888)))
        return image
    
    def show_image_from_array(self, image, ori=False, det=False):
        resize_image = cv2.cvtColor(resize_img(image), cv2.COLOR_RGB2BGR)
        if ori:
            self.ui.label_ori.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(resize_image.data, resize_image.shape[1], resize_image.shape[0], QtGui.QImage.Format_RGB888)))
        if det:
            self.ui.label_det.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(resize_image.data, resize_image.shape[1], resize_image.shape[0], QtGui.QImage.Format_RGB888)))
    
    def show_message(self, message):
        QMessageBox.information(self, "提示", message, QMessageBox.Ok)
    
    def reset_timer(self):
        self._timer.stop()
        self._timer = None
    
    def reset_timer_video(self):
        self._timer_video.stop()
        self._timer_video = None
    
    def reset_video_count(self):
        if self.video_count is not None:
            self.video_count = None
    
    def reset_det_label(self):
        self.ui.label_det.setText('')
    
    def select_model(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选取文件', '.', 'PT (*.pt);;ONNX (*.onnx)')
        self.ui.textBrowser.append(f'load model form {fileName}.')
        if fileName != '':
            # read cfg
            with open('model.yaml') as f:
                cfg = yaml.load(f, Loader=yaml.SafeLoader)
            # update model_path
            cfg['model_path'] = fileName
            # init yolov7 model
            self.model = yolov7(**cfg)
            self.ui.textBrowser.append(f'load model success.')
        else:
            self.ui.textBrowser.append(f'load model failure.')
            self.show_message('请选择模型文件.')
    
    def select_image_file(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选取文件', '.', 'JPG (*.jpg);;PNG (*.png)')
        if fileName != '':
            self.reset_det_label()
            image = self.read_and_show_image_from_path(fileName)
            self.now = image
            self.ui.textBrowser.append(f'read image form {fileName}')
        else:
            self.show_message('请选择图片文件.')

    def select_folder_file(self):
        folder = QFileDialog.getExistingDirectory(self, '选择路径', '.')
        folder_list = [os.path.join(folder, i) for i in os.listdir(folder)]
        if len(folder_list) == 0:
            self.show_message('选择的文件夹内容为空.')
        else:
            self.reset_det_label()
            self.now = folder_list
            self.read_and_show_image_from_path(folder_list[0])
            self.ui.textBrowser.append(f'read folder form {folder}')
    
    def select_video_file(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选取文件', '.', 'MP4 (*.mp4)')
        cap = cv2.VideoCapture(fileName)
        
        if self._timer is not None:
            self.reset_timer()
        
        if not cap.isOpened():
            self.show_message('视频打开失败.')
        else:
            self.reset_det_label()
            flag, image = cap.read()
            self.show_image_from_array(image, ori=True)
            self.now = cap
            self.video_count = int(self.now.get(cv2.CAP_PROP_FRAME_COUNT))
            self.print_id = 1

    def select_camear(self):
        self.show_camera_flag = 1
        self.show_video_only_flag = 0
        # self.cap.release()
        self.cap = cv2.VideoCapture(0)
        
        if self._timer is not None:
            self.reset_timer()
        
        if not self.cap.isOpened():
            self.show_message('视频打开失败.')
        else:
            self.print_id = 1
            self.reset_det_label()
            flag, image = self.cap.read()
            self.show_image_from_array(image, ori=True)
            self.now = self.cap
        self.begin_detect()
    
    def begin_detect(self):
        if self.model is None:
            self.show_message('请先选择模型.')
            
        if self._timer is not None:
            self.reset_timer()
        
        if self._timer_video is not None:
            self.reset_timer_video()
        
        if type(self.now) is cv2.VideoCapture:
            # fourcc  = cv2.VideoWriter_fourcc(*'mp4v') #XVID
            # size    = (int(self.now.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.now.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # self.out = cv2.VideoWriter(os.path.join(self.save_path, f'{self.save_id}.mp4'), fourcc, 25.0, size)
            
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.show_video)
            self._timer.start(40)
        elif type(self.now) is list:
            self.print_id, self.folder_len = 1, len(self.now)
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.show_folder)
            self._timer.start(20)
        else:
            image_det, label, conf = self.model(self.now)
            cv2.imwrite(os.path.join(self.save_path, f'{self.save_id}.jpg'), image_det)
            self.ui.textBrowser.append(f'save image in {os.path.join(self.save_path, f"{self.save_id}.jpg")}')
            self.save_id += 1
            self.show_image_from_array(image_det, ori=True)

            for i in range(len(label)):
                item = QtWidgets.QTableWidgetItem('%s'%(label[i]))
                # item = QStandardItem('%s'%(label[i]))  # label
                item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
                self.ui.tableWidget.setItem(self.save_id-1+i,0,item)

                item = QtWidgets.QTableWidgetItem('%s'%(1))
                # item = QStandardItem('%s'%(1))    #number
                item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
                self.ui.tableWidget.setItem(self.save_id-1+i,1,item)

                item = QtWidgets.QTableWidgetItem('%s'%(conf[i]))
                # item = QStandardItem('%s'%(conf[i]))
                item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
                self.ui.tableWidget.setItem(self.save_id-1+i,2,item)
    
    def show_video_only(self):
        self.show_video_only_flag = 1
        self.show_camera_flag = 0
        fileName = 'video/testv.mp4'
        self.cap = cv2.VideoCapture(fileName)
        
        if self._timer_video is not None:
            self.reset_timer_video()
        
        if not self.cap.isOpened():
            self.show_message('视频打开失败.')
        else:
            self.reset_det_label()
            flag, image = self.cap.read()
            self.show_image_from_array(image, ori=True)
            self.now = self.cap
            self.video_count = int(self.now.get(cv2.CAP_PROP_FRAME_COUNT))
            self.print_id = 1
        # fourcc  = cv2.VideoWriter_fourcc(*'mp4v') #XVID
        # size    = (int(self.now.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.now.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # self.out = cv2.VideoWriter(os.path.join(self.save_path, f'{self.save_id}.mp4'), fourcc, 25.0, size)
        
        self._timer_video = QTimer(self)
        self._timer_video.timeout.connect(self._show_video_only)
        self._timer_video.start(40)
    
    def stop_detect(self):
        if self._timer is not None:
            self.reset_timer()
            self.reset_video_count()
    
    def select_savepath(self):
        folder = QFileDialog.getExistingDirectory(self, '选择路径', '.')
        self.save_path = folder
        self.save_id = 0
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    
    def show_folder(self):
        if len(self.now) == 0:
            self.reset_timer()
        else:
            img_path = self.now[0]
            image = self.read_and_show_image_from_path(img_path)
            image_det = self.model(image)
            cv2.imwrite(os.path.join(self.save_path, f'{self.save_id}.jpg'), image_det)
            self.ui.textBrowser.append(f'{self.print_id}/{self.folder_len} save image in {os.path.join(self.save_path, f"{self.save_id}.jpg")}')
            self.show_image_from_array(image_det, det=True)
            self.print_id += 1
            self.save_id += 1
            self.now.pop(0)
    
    def show_video(self):
        start = time.time()
        flag, image = self.cap.read()
        # flag, image = self.now.read()
        # if flag:
        # self.show_image_from_array(image, ori=True)
        image_det, label, conf = self.model(image)
        if label == []:
            print('frame empty')
            pass
        else:
            self.label_all[self.label_count].append(label)
            self.label_count += 1
            # self.out.write(image_det)
            self.show_image_from_array(image_det, ori=True)
            if self.video_count is not None:
                self.ui.textBrowser.append(f'{self.print_id}/{self.video_count} Frames.')
            else:
                self.ui.textBrowser.append(f'{self.print_id} Frames.')
            self.print_id += 1
            # for i in range(len(label)):
            #     item = QtWidgets.QTableWidgetItem('%s'%(label[i]))
            #     # item = QStandardItem('%s'%(label[i]))  # label
            #     item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
            #     self.ui.tableWidget.setItem(self.print_id-1+i,0,item)

            #     item = QtWidgets.QTableWidgetItem('%s'%(1))
            #     # item = QStandardItem('%s'%(1))    #number
            #     item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
            #     self.ui.tableWidget.setItem(self.print_id-1+i,1,item)

            #     item = QtWidgets.QTableWidgetItem('%s'%(conf[i]))
            #     # item = QStandardItem('%s'%(conf[i]))
            #     item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
            #     self.ui.tableWidget.setItem(self.print_id-1+i,2,item)
        # else:
        #     self.now = None
        #     self.reset_timer()
        #     # self.out.release()
        #     self.reset_video_count()
        #     self.save_id += 1
        if self.label_count == 5:
            result = all_np(self.label_all)
            print(result)
            if result:
                if result[label[0]] >= 5:
                    self.reset_timer()
                    print('start dropping')
                    self.label_all = [[],[],[],[],[]]
                    self.label_count = 0
                    # 进行舵机投放
                    cat = tran(label[0])
                    disposal(cat)

                
                    item = QtWidgets.QTableWidgetItem('%s'%(label[0]))
                    # item = QStandardItem('%s'%(label[i]))  # label
                    item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
                    self.ui.tableWidget.setItem(self.obj_cont-1,0,item)

                    item = QtWidgets.QTableWidgetItem('%s'%(1))
                    # item = QStandardItem('%s'%(1))    #number
                    item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
                    self.ui.tableWidget.setItem(self.obj_cont-1,1,item)

                    item = QtWidgets.QTableWidgetItem('%s'%(conf[0]))
                    # item = QStandardItem('%s'%(conf[i]))
                    item.setTextAlignment(Qt.AlignCenter)      # 设置文本居中
                    self.ui.tableWidget.setItem(self.obj_cont-1,2,item)

                    self.obj_cont += 1

                    while 1:
                        if self.multily_drop == 1 and self.drop_1 == 0:
                            # 进行1舵机开门
                            print('start openning door one')
                            self.get_ready = 0

                            self.sensor_cont == 300
                            self.drop_1 = 1
                            break
                        if self.multily_drop == 1 and self.drop_2 == 0:
                            # 进行2舵机开门
                            print('start openning door two')
                            self.get_ready = 0

                            self.sensor_cont == 300
                            self.drop_2 = 1
                            break
                        if self.multily_drop == 1 and self.drop_3 == 0:
                            # 进行3舵机开门
                            print('start openning door three')
                            self.sensor_cont == 300
                            self.drop_3 = 1
                            break
                        if self.drop_1 == 1 and self.drop_2 == 1 and self.drop_3 == 1:
                            print('reset door')
                            self.drop_1 = 0
                            self.drop_2 = 0
                            self.drop_3 = 0
                            break
                        if self.multily_drop == 0:
                            break

                else:
                    for i in range(4):
                        self.label_all[i] = self.label_all[i+1]
                    self.label_count = 4
                    self.label_all[4] = []
            # self.cap.release()
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.show_video)
            self._timer.start(40)
            # fileName = 'video/testv.mp4'
            # self.cap = cv2.VideoCapture(fileName)
            # self._sensor = QTimer(self)
            # self._sensor.timeout.connect(self.signel_det)
            # self._sensor.start(20)
            # self._timer_video = QTimer(self)
            # self._timer_video.timeout.connect(self._show_video_only)
            # self._timer_video.start(40)
        end = time.time()
        print('Total cost: ',end - start, 's')
        # self.select_camear()
    
    def _show_video_only(self):
        flag, image = self.cap.read()
        # flag, image = self.now.read()
        if flag:
            # self.show_image_from_array(image, ori=True)
            # image_det = self.model(image)
            image_det = image
            # self.out.write(image_det)
            self.show_image_from_array(image_det, ori=True)
            if self.video_count is not None:
                self.ui.textBrowser.append(f'{self.print_id}/{self.video_count} Frames.')
            else:
                self.ui.textBrowser.append(f'{self.print_id} Frames.')
            self.print_id += 1
        else:
            self.now = None
            self.reset_timer_video()
            # self.out.release()
            self.reset_video_count()
            self.show_video_only()
            self.save_id += 1
    
    def manzai(self):
        palette_red = QPalette()
        palette_red.setColor(QPalette.Window, Qt.red)
        self.ui.bin_kehuishou.setPalette(palette_red)

    def _exit(self):
        self.close()

if __name__ == '__main__':
    gui_title = 'Yolo-MaskDet'
    
    app = QApplication(sys.argv)
    w = MyForm(title='')
    sys.exit(app.exec_())
