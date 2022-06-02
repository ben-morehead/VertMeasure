from distutils.command.upload import upload
import os
import sys
import time
import logging
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QLineEdit, QPushButton,
                             QStackedLayout, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
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
        self.ch = None
        self.__calibration_qImg = None

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
        self.config_p = self.config_page_generator()
        self.cal_scale = None
        self.export_p = None
        self.demo = None
        
        self.stackedLayout.addWidget(self.entrance)
        self.stackedLayout.addWidget(self.config_p)
        
        self.general_layout.addLayout(self.stackedLayout)
        print(f"Stacked Layout Count: {self.stackedLayout.count()}")

        self.showFullScreen()

    '''----- Program Infrastructure ----- '''

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
            self.stackedLayout.removeWidget(self.cal_scale)
            self.stackedLayout.removeWidget(self.export_p)
            self.stackedLayout.removeWidget(self.demo)
            self.ch = None
            self.stackedLayout.setCurrentIndex(1)
            return

        self.stackedLayout.setCurrentIndex(self.stackedLayout.currentIndex() + 1)
        self.log.info(f"Now Switching Pages from {self.stackedLayout.currentIndex()} to {self.stackedLayout.currentIndex() + 1}")
    
    def reset_page(self):
        self.reset_program = 1
        self.next_page()

    '''----- Output Formatting -----'''

    def export_jump_info(self):
        output_path = ".\\info_exports"
        self.ch.export_jump_info()
        self.export_button_label.setText("Information succesfully exported to the output folder")
        self.export_button_label.update()
        
    
    '''----- UI Definitions ------'''

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

    def config_page_generator(self):
            config_page = QWidget()
            config_layout = QVBoxLayout()
            button_layout = QHBoxLayout()
            
            ###Entry Definitions

            # Video Source: Widget declaration --------------------------------------------
            upload_entry = QHBoxLayout()
            upload_label = QLabel("Video Path:")
            self.upload_line = QLineEdit()
            upload_button = QPushButton("Upload", clicked=self.get_video_file)
            # Widget Specifications
            self.upload_line.setMinimumWidth(self.windowWidth * 0.6)
            self.upload_line.setReadOnly(1)
            upload_button.setMinimumWidth(self.windowWidth * 0.2)
            # Layout
            upload_entry.addWidget(upload_label)
            upload_entry.addStretch(1)
            upload_entry.addWidget(self.upload_line)
            upload_entry.addStretch(1)
            upload_entry.addWidget(upload_button)

            # Name: Widget declaration ----------------------------------------------------
            name_entry = QHBoxLayout()
            name_label = QLabel("Name:")
            self.name_line = QLineEdit()
            # Widget Specifications
            self.name_line.setMinimumWidth(self.windowWidth * 0.9)
            # Layout
            name_entry.addWidget(name_label)
            name_entry.addStretch(1)
            name_entry.addWidget(self.name_line)

            # Height: Widget declaration --------------------------------------------------
            height_entry = QHBoxLayout()
            height_label = QLabel("Height (inches):")
            self.height_line = QLineEdit()
            # Widget Specifications
            self.height_line.setMinimumWidth(self.windowWidth * 0.8)
            # Layout
            height_entry.addWidget(height_label)
            height_entry.addStretch(1)
            height_entry.addWidget(self.height_line)

            # Jump Style: Widget declaration -------------------------------------------------
            style_entry = QHBoxLayout()
            style_label = QLabel("Reference Point:")
            self.style_ground = QRadioButton("Ground")
            self.style_rim = QRadioButton("Rim")
            
            self.style_ground.setChecked(1)
            # Widget Specifications
            self.style_ground.setMinimumWidth(self.windowWidth * 0.1)
            self.style_rim.setMinimumWidth(self.windowWidth * 0.1)
            # Layout
            style_entry.addWidget(style_label)
            style_entry.addStretch(1)
            style_entry.addWidget(self.style_ground)
            style_entry.addWidget(self.style_rim)
            style_entry.addStretch(5)
            



            #Button Definitions
            self.config_set_msg = QLabel("")
            self.config_set_button = QPushButton("Confirm", clicked=self.confirm_config)
            self.config_set_button.setFixedWidth(self.windowWidth / 5)
            
            button_layout.addWidget(self.config_set_msg)
            button_layout.addWidget(self.config_set_button, alignment=Qt.AlignmentFlag.AlignRight)
            config_layout.addStretch(2)
            config_layout.addLayout(upload_entry)
            config_layout.addStretch(1)
            config_layout.addLayout(name_entry)
            config_layout.addStretch(1)
            config_layout.addLayout(height_entry)
            config_layout.addStretch(1)
            config_layout.addLayout(style_entry)
            config_layout.addStretch(12)
            config_layout.addLayout(button_layout)
            config_page.setLayout(config_layout)

            return config_page

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
    
    def export_page_generator(self):
        export_page = QWidget()
        export_layout = QVBoxLayout()
        button_layout = QVBoxLayout()
        button_layout_row_a = QHBoxLayout()
        button_layout_row_b = QHBoxLayout()

        self.export_label = QLabel(text=f"Vertical Jump: {self.measured_jump_height:.2f} inches")
        self.export_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.demo_btn = QPushButton("See Demo", clicked=self.setup_demo_page)
        self.export_btn = QPushButton("Export Information", clicked=self.export_jump_info)
        self.export_exit_button = QPushButton("Exit Application", clicked=self.close)
        self.export_reset_button = QPushButton("Calculate New Jump", clicked=self.reset_page)
        self.export_button_label = QLabel("")
        self.export_button_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        button_layout_row_a.addWidget(self.demo_btn)
        button_layout_row_a.addWidget(self.export_btn)
        button_layout_row_b.addWidget(self.export_reset_button)
        button_layout_row_b.addWidget(self.export_exit_button)
        button_layout.addWidget(self.export_button_label)
        button_layout.addLayout(button_layout_row_a)
        button_layout.addLayout(button_layout_row_b)

        export_layout.addStretch(1)
        export_layout.addWidget(self.export_label)
        export_layout.addStretch(1)
        export_layout.addLayout(button_layout)
        export_page.setLayout(export_layout)
    
        return export_page

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

    '''----- Config Page Helpers ----- '''

    def confirm_config(self):
        valid_set = True
        upload_name = self.upload_line.text()
        jumper_name = self.name_line.text()
        jumper_height = self.height_line.text()
        jump_style = -1 + (1 * self.style_ground.isChecked()) + (2 * self.style_rim.isChecked())

        if upload_name == "" or jumper_name == "" or jumper_height=="":
            self.log.info("TEXT FIELDS NOT ENTERED")
            valid_set = False

        if valid_set:
            jumper_height = float(jumper_height)
            self.log.info(f"Upload Path: {upload_name}")
            self.log.info(f"Jumper Name: {jumper_name}")
            self.log.info(f"Jumper Height: {jumper_height}")
            self.log.info(f"Jump Style Index: {jump_style}")
            
            #Calibration Handler Setup
            self.ch = CalibrationHandler(source_name=upload_name, jumper_name=jumper_name, jumper_height=jumper_height, jump_style=jump_style, log=self.log)
            self.ch.generate_video_points()
            self.ch.define_joint_averages()
            
            self.ch.define_stages()
            self.ch.get_reference_values()
            
            self.ch.estimate_head_height()

            self.cal_scale = self.calibration_page_generator()
            self.export_p = self.export_page_generator()
            self.demo = self.demo_page_generator()

            self.stackedLayout.addWidget(self.cal_scale)
            self.stackedLayout.addWidget(self.export_p)
            self.stackedLayout.addWidget(self.demo)
            
            self.next_page()

        else:
            self.config_set_msg.setText("Error: Ensure all configuration settings are complete")

    def get_video_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
            '.\\vid_src',"Video Files (*.mov *.mp4 *.avi)")
        self.upload_line.clear()
        self.upload_line.insert(fname[0])

    '''----- Calibration Page Helpers ----- '''

    def increase_shoulder_offset(self):
        self.shoulder_offset -= 1
        self.update_calibration_img()

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

    '''----- Demo Page Helpers ----- '''
    
    def setup_demo_page(self):
        self.ch.setup_demo()
        self.demo_timer.start(1000/self.frame_rate_demo)
        self.next_page()
    
    def update_demo_display(self):
        vert, frame = np.copy(self.ch.get_demo_frame())
        frame_img = Image.fromarray(frame)
        self.__demo_qImg = ImageQt(frame_img)
        self.kin_pixmap = QPixmap.fromImage(self.__demo_qImg)
        self.demo_label.setPixmap(self.kin_pixmap)
        self.vertical_label.setText(f"Current Height: {vert}")
        self.demo_label.update()

    def reset_demo_page(self):
        self.demo_timer.stop()
        self.demo_time = None
        self.reset_page()

    '''----- DEPRECATED BUT USEFUL ----- '''

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

