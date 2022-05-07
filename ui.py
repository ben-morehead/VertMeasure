import os
import sys
import time
import logging
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QLineEdit, QPushButton,
                             QStackedLayout, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
                             QLabel)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
import numpy as np

from handlers import CalibrationHandler
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2 

sys.path.append('D:\\Basketball\\VertMeasure')

class Window(QWidget):
    def __init__(self, screen_width, screen_height):
        super().__init__()
        self.setup_logger()
        self.ch = CalibrationHandler()
        self.__calibration_qImg = None

        #Calibration Handler Setup
        self.ch.define_stages()
        self.ch.get_stage_0_vals()
        self.ch.estimate_head_height()

        #DEV STUFF
        self.toggle = True
        self.shoulder_offset = 0
        self.mouse_state = 0
        self.border_offset = 12
        self.measured_jump_height = -1
        self.cal_tl, self.cat_br = (0, 11), (1897, 1068)

        #Timers
        self.frame_rate_demo = 15
        self.demo_timer = None

        #General Application Setup
        self.reset_program = 0
        self.app_start_time = time.time()
        self.setWindowTitle("Vertical Jump Measurement")
        self.windowHeight, self.windowWidth = screen_height, screen_width
        self.setFixedSize(self.windowWidth, self.windowHeight)
        
        #Page Layout Setup
        self.general_layout = QVBoxLayout()
        self.setLayout(self.general_layout)
        self.stackedLayout = QStackedLayout()
        self.entrance = self.entrance_page_generator()
        self.cal_scale = self.calibration_page_generator()
        self.export_p = self.export_page_generator()
        self.demo = self.demo_page_generator()

        self.stackedLayout.addWidget(self.entrance)
        self.stackedLayout.addWidget(self.cal_scale)
        self.stackedLayout.addWidget(self.export_p)
        self.stackedLayout.addWidget(self.demo)
        self.general_layout.addLayout(self.stackedLayout)
        print(f"Stacked Layout Count: {self.stackedLayout.count()}")

        self.showFullScreen()

    def setup_logger(self):
        self.log = logging.getLogger('VertMeasure')
        self.log.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('run.log', mode="w")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.log.addHandler(fh)
        self.log.addHandler(ch)

    def keyPressEvent(self, e):
        #Closes the application on the window event e
        if e.key() == Qt.Key_Escape:
            self.log.info("Exiting Program Via: Escape Key")
            self.close()
    
    def next_page(self):
        if self.reset_program:
            self.reset_program = 0
            self.stackedLayout.setCurrentIndex(0)
            self.export_jump_info()
            return

        self.stackedLayout.setCurrentIndex(self.stackedLayout.currentIndex() + 1)
        self.log.info(f"Now Switching Pages from {self.stackedLayout.currentIndex()} to {self.stackedLayout.currentIndex() + 1}")
    
    def reset_page(self):
        self.reset_program = 1
        self.next_page()

    def export_jump_info(self):
        print("TO IMPLEMENT: Export the Jumping Statistics")
    
    def calibration_page_generator(self):
        calibration_page = QWidget()
        calibration_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.calibration_label = QLabel(self)
        #self.calibration_label = QLabel(str(self.shoulder_offset))
        self.calibration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shoulder_offset_btn = QPushButton("Increase", clicked=self.increase_shoulder_offset)
        self.sub_shoulder_offset_btn = QPushButton("Decrease", clicked=self.decrease_shoulder_offset)
        self.confirm_shoulder_offset_btn = QPushButton("Confirm", clicked=self.confirm_shoulder_offset)

        button_layout.addWidget(self.add_shoulder_offset_btn)
        button_layout.addWidget(self.sub_shoulder_offset_btn)
        button_layout.addWidget(self.confirm_shoulder_offset_btn)

        calibration_layout.addWidget(self.calibration_label)
        calibration_layout.addLayout(button_layout)
        calibration_page.setLayout(calibration_layout)
         
        #Loading the loading calibration image
        init_frame = self.ch.get_init_head_frame()
        frame_img = Image.fromarray(init_frame)
        self.__calibration_qImg = ImageQt(frame_img)
        self.kin_pixmap = QPixmap.fromImage(self.__calibration_qImg)
        self.calibration_label.setPixmap(self.kin_pixmap)
        

        return calibration_page
    
    def increase_shoulder_offset(self):
        self.shoulder_offset -= 1
        self.update_calibration_img()
        #self.calibration_label.setText(str(self.shoulder_offset))

    def decrease_shoulder_offset(self):
        self.shoulder_offset += 1
        self.update_calibration_img()

    def confirm_shoulder_offset(self):
        self.ch.calibrate_head_height(self.shoulder_offset)
        self.measured_jump_height = self.ch.calculate_vertical_jump()
        self.export_label.setText(f"Vertical Jump: {self.measured_jump_height:.2f} inches")
        self.next_page()
    
    def update_calibration_img(self):
        init_frame = self.ch.get_adjusted_head_frame(self.shoulder_offset)
        frame_img = Image.fromarray(init_frame)
        self.__calibration_qImg = ImageQt(frame_img)
        self.kin_pixmap = QPixmap.fromImage(self.__calibration_qImg)
        self.calibration_label.setPixmap(self.kin_pixmap)
        self.calibration_label.update()

    def export_page_generator(self):
        export_page = QWidget()
        export_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.export_label = QLabel(text=f"Vertical Jump: {self.measured_jump_height:.2f} inches")
        self.export_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.demo_btn = QPushButton("See Demo", clicked=self.setup_demo_page)
        self.export_btn = QPushButton("Export Information", clicked=self.reset_page)

        button_layout.addWidget(self.demo_btn)
        button_layout.addWidget(self.export_btn)

        export_layout.addWidget(self.export_label)
        export_layout.addLayout(button_layout)
        export_page.setLayout(export_layout)
    
        return export_page
    
    def setup_demo_page(self):
        self.ch.setup_demo()
        self.demo_timer.start(1000/self.frame_rate_demo)
        self.next_page()

    def reset_demo_page(self):
        self.reset_page()

    def demo_page_generator(self):
        demo_page = QWidget()
        demo_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.vertical_label = QLabel(f"Current Height: {0}")
        self.exit_button = QPushButton("Exit Application", clicked=self.close)
        self.demo_reset_button = QPushButton("Calculate New Jump", clicked=self.reset_demo_page)
        self.demo_label = QLabel()
        self.demo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        button_layout.addWidget(self.demo_reset_button)
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.vertical_label)
        demo_layout.addWidget(self.demo_label)
        demo_layout.addLayout(button_layout)
        demo_page.setLayout(demo_layout)
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self.update_demo_display)
        return demo_page

    def update_demo_display(self):
        vert, frame = np.copy(self.ch.get_demo_frame())
        frame_img = Image.fromarray(frame)
        self.__demo_qImg = ImageQt(frame_img)
        self.kin_pixmap = QPixmap.fromImage(self.__demo_qImg)
        self.demo_label.setPixmap(self.kin_pixmap)
        self.vertical_label.setText(f"Current Height: {vert}")
        self.demo_label.update()

    def entrance_page_generator(self):
        entrance_page = QWidget()
        
        entrance_layout = QVBoxLayout()
        self.entrance_label = QLabel("Vertical Jump Measurement")
        self.entrance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.begin_button = QPushButton("BEGIN", clicked=self.next_page)
        entrance_layout.addWidget(self.entrance_label)
        entrance_layout.addWidget(self.begin_button)
        entrance_page.setLayout(entrance_layout)

        return entrance_page

    def console_maker(self):
        logOutput = QTextEdit()
        logOutput.setReadOnly(True)
        logOutput.setLineWrapMode(QTextEdit.NoWrap)

        font = logOutput.font()
        font.setFamily("Courier")
        font.setPointSize(10)
        return logOutput

    def ground_maker(self):
        ground_widget = QWidget()
        self.ground_line = QLineEdit()
        self.ground_console = self.console_maker()
        self.upload_btn = QPushButton("Upload Order", clicked=self.getfile)
        self.clear_btn = QPushButton("&CLEAR", clicked=self.ground_console.clear)
        self.ground_line.returnPressed.connect(self.on_line_edit_enter)
        
        
        upper_layout = QVBoxLayout()
        layout_top = QVBoxLayout()
        layout_bot = QHBoxLayout()
        
        layout_top.addWidget(self.ground_console)
        layout_bot.addWidget(self.ground_line, 7)
        layout_bot.addWidget(self.upload_btn, 2)
        layout_bot.addWidget(self.clear_btn, 1)
        

        upper_layout.addLayout(layout_top)
        upper_layout.addLayout(layout_bot)

        ground_widget.setLayout(upper_layout)
        return ground_widget

    def on_line_edit_enter(self):
        line_value = self.ground_line.text()
        self.ground_line.clear()
        ret = self.sp.import_sheet(line_value)
        if ret:
            self.ground_console.append("Succesfully Imported Spreadsheet")
        data_str = self.sp.display_data()
        self.ground_console.append(data_str)
        self.sp.parse_data()
        valid_check = self.sp.check_valid()
        self.ground_console.append(valid_check[1])
        if valid_check[0]:
            encoding = self.sp.encode_to_packet()
            self.ground_console.append(encoding)
    
    def hospital_maker(self):
        
        hospital_widget = QWidget()
        self.hospital_start_button = QPushButton("START", clicked=self.start_hospital_loop)
        self.hospital_stop_button = QPushButton("STOP", clicked=self.end_hospital_loop)
        self.hospital_console = self.console_maker()
        self.hospital_stop_button.setEnabled(False)
        
        upper_layout = QVBoxLayout()
        layout_top = QVBoxLayout()
        layout_bot = QHBoxLayout()
        
        layout_top.addWidget(self.hospital_console)
        layout_bot.addWidget(self.hospital_start_button, 5)
        layout_bot.addWidget(self.hospital_stop_button, 5)

        upper_layout.addLayout(layout_top)
        upper_layout.addLayout(layout_bot)

        hospital_widget.setLayout(upper_layout)
        return hospital_widget

    def end_hospital_loop(self):
        self.hospital_timer.stop()
        del self.hospital_timer
        self.hospital_console.append("Hospital Loop Ended")

        self.hospital_start_button.setEnabled(True)
        self.hospital_stop_button.setEnabled(False)

