from tracemalloc import start
import mediapipe as mp
import cv2 as cv
import time
import numpy as np
import mapping
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import csv



class PoseHandler():
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=0,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence= 0.5,
                 min_tracking_confidence = 0.95):
        '''
        Initialize the poseDetector
        '''

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)


    def findPose(self, img, draw = True):
        '''
        findPose takes in the img you want to find the pose in, and whether or not you
        want to draw the pose (True by default).
        '''
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) #convert BGR --> RGB as openCV uses BGR but mediapipe uses RGB
        imgRGB.flags.writeable = False
        self.results = self.pose.process(image = imgRGB)
        if self.results.pose_landmarks and draw: # if there are pose landmarks in our results object and draw was set to True
            self.mpDraw.draw_landmarks(imgRGB, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color = (255, 255, 255), thickness = 2, circle_radius = 2), self.mpDraw.DrawingSpec(color = (35, 176, 247), thickness = 2, circle_radius = 2)) #draw them

        return imgRGB #returns the image

    
    def findPosition(self, img, lm_select, draw=True):
        '''
        Takes in the image and returns a list of the landmarks
        '''
        lm_conversions = []
        for element in lm_select: lm_conversions.append(mapping.landmarks[element]) 
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark): #for each landmark
                if id in lm_conversions:
                    h, w, c = img.shape #grab the image shape
                    cx, cy = int(lm.x * w), int(lm.y * h) # set cx and cy to the landmark coordinates
                    if cx < 1920 and cy < 1080 and cx > 0 and cy > 0:
                        lmList.append([id, cx, cy]) #append the landmark id, the x coord and the y coord
                    if draw:
                        cv.circle(img, (cx,cy), 10, (255, 0, 0), cv.FILLED) #if we want to draw, then draw
        if draw: return lmList, img
        return lmList #returns the list of landmarks


    def findAngle(self, p1, p2, p3):
        '''
        takes in three points and returns the angle between them
        '''
        self.p1 = np.array(p1) # start point
        self.p2 = np.array(p2) # mid point
        self.p3 = np.array(p3) # end point
        
        radians = np.arctan2(self.p3[2]-self.p2[2], self.p3[1]-self.p2[1]) - np.arctan2(self.p1[2]-self.p2[2], self.p1[1]-self.p2[1]) #trig
        angle = np.abs(radians*180.0/np.pi) 
        
        if angle >180.0:
            angle = 360-angle
            
        return angle
    
    def get_shoulder_value(self, frame):
        self.findPose(frame, draw = False)
        values = self.findPosition(frame, ["left_shoulder", "right_shoulder"], draw=False)
        return values

class CalibrationHandler():
    def __init__(self, source_name="vid_src\\ben_2.mov", jumper_name="Ben", jumper_height=72, jump_style=4, log=logging.getLogger(__name__)):
        '''
        Initializing the handler object
        '''

        
        self.source_name = source_name
        self.log = log
        self.base_frame_tolerance = 1
        #self.source_name = "No_Strength_Shortening_1.mp4"
        #self.source_name = "ben_2.mov"
        self.jumper_name= jumper_name
        self.jump_style = jump_style

        self.log.info("Calibration Handler Created")
        
        self.base_frame = None
        
        #self.height_measured = 77.5 #CONFIGURATION VALUE
        self.height_measured = jumper_height
        self.stage_tolerance = 3 #HYPERPARAMETER
        self.num_max_vals = 1 #HYPERPARAMETER
        self.shoulder_head_ratio_estimate = 1.25 #HYPERPARAMETER
        self.base_fps = 30

        self.shoulder_levels = []

        self.inch_vert = 0
        self.head_point = 500
        self.pose_handler = PoseHandler()
        self.stage_split = 0
        self.pixels_per_inch = 5
        

        #NOTE: These should happen whenever the video changes i.e. the system resets
        #self.generate_video_points()
        #self.define_joint_averages()

    def convert_joint_index_to_label(self, index):
        '''
        Converts the joint index for mediapipe into its corresponding label
        '''
        return list(mapping.landmarks.keys())[list(mapping.landmarks.values()).index(index)]

    def convert_joint_label_to_index(self, label):
        '''
        Converts the joint label for mediapipe into its corresponding index value
        '''
        return mapping.landmarks[label]

    def generate_video_points(self):
        '''
        Generates a dictionary (self.video_joint_dict) that contains all the needed data from mediapipe of the jump video
        '''
        self.video_joint_dict = {}
        cap = cv.VideoCapture(f"{self.source_name}")

        count = 0
        while True:
            ret, frame = cap.read()
            if frame is not None:
                joint_set = [11, 12, 23, 24, 29, 30, 31, 32]
                pose_frame = np.copy(frame)
                self.pose_handler.findPose(pose_frame, draw = False)
                all_values = self.pose_handler.findPosition(pose_frame, ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_heel", "right_heel", "left_foot_index", "right_foot_index"], draw=False)
                temp_joint_set = []
                for value in all_values:
                    if count==0:
                        self.video_joint_dict[self.convert_joint_index_to_label(value[0])] = []
                    self.video_joint_dict[self.convert_joint_index_to_label(value[0])].append((count, value[1], value[2]))
                    temp_joint_set.append(value[0])
                difference_joint_set = set(joint_set).symmetric_difference(set(temp_joint_set))
                for value in difference_joint_set:
                    if count==0:
                        self.video_joint_dict[self.convert_joint_index_to_label(value)] = []
                    self.video_joint_dict[self.convert_joint_index_to_label(value)].append((count, -1, -1))
            else:
                break
            count+=1
        
        self.frame_count = count
        cap.release()
        cv.destroyAllWindows()
        self.log.info(f"generate_video_points(self) | Dictionary of Joints Created with Keys: {list(self.video_joint_dict.keys())}")

    
    def setup_demo(self):
        '''
        Initializing the opencv video capture device for the demonstration section
        '''
        self.demo_vidcap = cv.VideoCapture(f"{self.source_name}")
        self.demo_frame_count = 0

    def get_demo_frame(self):
        '''
        Prepares and provides the frame for the demonstration section of the UI
        '''
        ret, frame = self.demo_vidcap.read()
        if frame is None:
            self.demo_frame_count = 0
            self.demo_vidcap.release()
            self.demo_vidcap = cv.VideoCapture(f"{self.source_name}")
            ret, frame = self.demo_vidcap.read()
        vert, frame = self.draw_demo_frame(frame=frame, frame_num=self.demo_frame_count)
        ret_frame = self.convert_to_formatted_frame(frame=frame)
        self.demo_frame_count += 1
        return vert, ret_frame

    def get_raw_base_frame(self):
        count = 0
        cap = cv.VideoCapture(f"{self.source_name}")
        while True:
            ret, frame = cap.read()
            if count == 5:
                self.base_frame = frame
                break
            if frame is None:
                break
            count+=1
        cap.release()

    def close_demo(self):
        '''
        Closes the demonstration video capture device
        '''
        self.demo_vidcap.release()
        self.demo_vidcap = None
        self.demo_frame_count = 0

    def export_jump_info(self):
        '''
        Exports the statistics from the jump to a graph and csv output file
        '''
        header = ["Name", "Maximum Vertical Jump (inches)", "Descent Speed (inches/s)", "Descent Level (inches)", "Ground Time (s)"]
        if self.jump_style == 0:
            time_id = datetime.now().strftime("%Y%m%d%H%M")
            name_base = f"{self.jumper_name}_{time_id}_two_foot"
            
            graph_x_vals = np.linspace(0,len(self.shoulder_levels),num=len(self.shoulder_levels),endpoint=False)
            graph_y_vals = np.array(self.shoulder_levels)
            graph_base_vals = np.array([int(((self.hal - self.hsl) + (self.har - self.hsr)) / 2)] * len(self.shoulder_levels))

            plt.xlabel('Time (Frames)')
            plt.ylabel('Height (Inches)')
            plt.plot(graph_x_vals, (graph_base_vals - graph_y_vals) / self.pixels_per_inch, 'o', color='red')
            plt.plot(graph_x_vals, graph_base_vals - graph_base_vals, '-', color='blue')
            plt.legend(["Shoulder Path", "Initial Shoulder Level"])
            plt.show()
            plt.savefig(f'info_exports\\{name_base}_graph.png')

            desc_level, desc_speed, ground_time = self.measure_descent_speed()

            data_row = [f"{self.jumper_name}", f"{self.inch_vert}", f"{desc_speed}", f"{desc_level}", f"{ground_time}"]

            with open(f'info_exports\\{name_base}_info.csv', 'w+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data_row)

        else:
            print("Jump Style not Implemented Yet")

    def measure_descent_speed(self):
        '''
        Measures the speed of the shoulder drop on the downwards action of the jump
        '''
        measure_descent = False
        measure_ground_time = False
        init_frame = 0
        descent_frame = 0
        init_level = 0
        lowest_point = 0
        final_ground_frame = 0

        for i, val in enumerate(self.shoulder_levels):
            if i == 0:
                prev_val = val
            else:
                prev_val = current_val
            current_val = val

            if measure_descent:
                if current_val < prev_val:
                    lowest_point = prev_val
                    descent_frame = (i - 1)
                    measure_descent = False
                    measure_ground_time = True
            else:
                if measure_ground_time:
                    if current_val > init_level:
                        final_ground_frame = i - 1
                        break
                else:
                    if current_val - prev_val > self.stage_tolerance:
                        init_level = prev_val
                        init_frame = (i - 1)
                        measure_descent = True

            
        self.log.info("measure_descent_speed(self) |ADVANCED DATA INFO")
        self.log.info("measure_descent_speed(self) |Init Frame = {init_frame}")
        self.log.info("measure_descent_speed(self) |Init Level = {init_level}")
        self.log.info("measure_descent_speed(self) |Descent Frame = {descent_frame}")
        self.log.info("measure_descent_speed(self) |Descent Level = {lowest_point}\n")
        self.log.info("measure_descent_speed(self) |Final Ground Frame = {final_ground_frame}")

        descent_level = (lowest_point - init_level) / self.pixels_per_inch
        descent_speed = descent_level / ((descent_frame - init_frame) / self.base_fps)
        ground_time = (final_ground_frame - init_frame) / self.base_fps
        self.log.info("measure_descent_speed(self) |Lowest Reached (inches): {descent_level:.2f}")
        self.log.info("measure_descent_speed(self) |Descent Speed (inches/second): {descent_speed:.2f}")
        self.log.info("measure_descent_speed(self) |Ground Time: {ground_time:.2f} seconds")
        return descent_level, descent_speed, ground_time
        
    def define_stages(self):
      
        split_found = 0
        count = 0
        #Code for displaying a video for 60fps
        #VIA VIDEO DICT

        self.get_raw_base_frame()

        for frame_number in range(0,self.frame_count):
            # Should be form (x,y)
            if frame_number == 0:
                current_shoulder_height = self.shoulder_averages[frame_number][1]
                prev_shoulder_height = current_shoulder_height
                
            else:
                prev_shoulder_height = current_shoulder_height
                current_shoulder_height = self.shoulder_averages[frame_number][1]
            
            frame_diff = abs(current_shoulder_height - prev_shoulder_height)
            
            if frame_diff > self.stage_tolerance and (not split_found):
                self.stage_split = frame_number
                split_found = 1
        self.log.info(f"define_stages(self) | Total Frame Count: {count}")
        self.log.info(f"define_stages(self) | Splitting Point: Frame {self.stage_split}")


        #VIA VIDEO CAP
        '''
        cap = cv.VideoCapture(f"{self.source_name}")
        while True:
            ret, frame = cap.read()
            if count == 5:
                self.base_frame = frame
            if frame is None:
                break

            pose_frame = np.copy(frame)
            shoulder_height = self.pose_handler.get_shoulder_value(pose_frame)
            avg_shoulder_height = 0
            for val in shoulder_height:
                avg_shoulder_height = avg_shoulder_height + val[2]
            avg_shoulder_height = avg_shoulder_height / len(shoulder_height)
            
            if count == 0:
                prev_shoulder_height = avg_shoulder_height
                current_shoulder_height = avg_shoulder_height
            else:
                prev_shoulder_height = current_shoulder_height
                current_shoulder_height = avg_shoulder_height

            frame_diff = abs(current_shoulder_height - prev_shoulder_height)
            if frame_diff > self.stage_tolerance and not split_found:
                self.stage_split = count
                split_found = 1
            count+=1
        
        cap.release()
        cv.destroyAllWindows()
        print(f"Total Frame Count: {count}")
        print(f"Splitting Point: Frame {self.stage_split}")
        '''

    def define_joint_averages(self):
        self.ankle_averages = []
        self.hip_averages = []
        self.shoulder_averages = []

        for frame_num in range(0, self.frame_count):
            ankle_check, hip_check, shoulder_check = self.check_for_joints(frame_num)
            #print(f"Frame Count: {frame_num}/{self.frame_count}")
            #print(f"Ankle Check, Hip Check, Shoulder Check = {ankle_check}, {hip_check}, {shoulder_check}")
            if ankle_check:
                left_foot_avg = ((self.video_joint_dict["left_heel"][frame_num][1] + self.video_joint_dict["left_foot_index"][frame_num][1])/2, (self.video_joint_dict["left_heel"][frame_num][2] + self.video_joint_dict["left_foot_index"][frame_num][2])/2)
                right_foot_avg = ((self.video_joint_dict["right_heel"][frame_num][1] + self.video_joint_dict["right_foot_index"][frame_num][1])/2, (self.video_joint_dict["right_heel"][frame_num][2] + self.video_joint_dict["right_foot_index"][frame_num][2])/2)
                self.ankle_averages.append((left_foot_avg, right_foot_avg))
            else:
                if frame_num == 0:
                    self.ankle_averages.append(0)
                else:
                    self.ankle_averages.append(self.ankle_averages[-1])

            if hip_check:
                hip_avg = ((self.video_joint_dict["left_hip"][frame_num][1] + self.video_joint_dict["right_hip"][frame_num][1])/2,(self.video_joint_dict["left_hip"][frame_num][2] + self.video_joint_dict["right_hip"][frame_num][2])/2)
            else:
                if frame_num == 0:
                    self.hip_averages.append(0)
                else:
                    self.hip_averages.append(self.hip_averages[-1])

            if shoulder_check:
                shoulder_avg = ((self.video_joint_dict["left_shoulder"][frame_num][1] + self.video_joint_dict["right_shoulder"][frame_num][1])/2,(self.video_joint_dict["left_shoulder"][frame_num][2] + self.video_joint_dict["right_shoulder"][frame_num][2])/2)
            else:
                if frame_num == 0:
                    self.shoulder_averages.append(0)
                else:
                    self.shoulder_averages.append(self.shoulder_averages[-1])

            #CAN ADD KNEE AVGS IN FUTURE VERSIONS
            #IF THE VALS ARENT -1 ADD EM
            #IF -1 APPEND THE PREVIOUS VALUE?
            
            self.hip_averages.append(hip_avg)
            self.shoulder_averages.append(shoulder_avg)

    def check_for_joints(self, frame_num):
        ankle_check, hip_check, shoulder_check = True, True, True
        if self.video_joint_dict["left_heel"][frame_num][1] == -1 or self.video_joint_dict["left_heel"][frame_num][2] == -1:
            ankle_check = False
        if self.video_joint_dict["right_heel"][frame_num][1] == -1 or self.video_joint_dict["left_heel"][frame_num][2] == -1:
            ankle_check = False
        if self.video_joint_dict["left_foot_index"][frame_num][1] == -1 or self.video_joint_dict["left_foot_index"][frame_num][2] == -1:
            ankle_check = False
        if self.video_joint_dict["right_foot_index"][frame_num][1] == -1 or self.video_joint_dict["right_foot_index"][frame_num][2] == -1:
            ankle_check = False
        if self.video_joint_dict["left_hip"][frame_num][1] == -1 or self.video_joint_dict["left_hip"][frame_num][2] == -1 or self.video_joint_dict["right_hip"][frame_num][1] == -1 or self.video_joint_dict["right_hip"][frame_num][2] == -1:
            hip_check = False
        if self.video_joint_dict["left_shoulder"][frame_num][1] == -1 or self.video_joint_dict["left_shoulder"][frame_num][2] == -1 or self.video_joint_dict["right_shoulder"][frame_num][1] == -1 or self.video_joint_dict["right_shoulder"][frame_num][2] == -1:
            shoulder_check = False
        return ankle_check, hip_check, shoulder_check
                
    def calculate_vertical_jump(self):
        if self.jump_style == 0:
            self.get_reference_jump_vals()
        else:
            print("JUMPING STYLE NOT YET IMPLEMENTED")
        pixel_vertical = int(((self.jsl-self.hsl) + (self.jsr-self.hsr))/2)
        inch_vertical = pixel_vertical / self.pixels_per_inch
        self.log.info(f"calculate_vertical_jump(self) | Vertical in Pixels: {pixel_vertical} | Vertical in Inches: {inch_vertical}")
        self.inch_vert = inch_vertical
        return inch_vertical

    def get_reference_values(self):
        """
        Retrieves H(ead) A(nkle) R(ight), HAL, HSL, and HSR, all of which are the base heights calculated from the shoulder to the original ankle level
        """
        frame_count = 0
        #cap = cv.VideoCapture(f"{self.source_name}")
        left_shoulder_list = []
        right_shoulder_list = []
        left_ankle_list = []
        right_ankle_list = []
        
        #FOR VIDEO DICT
        for frame_num in range(0, self.stage_split):
            #TO ACCOUNT FOR MISSING VALUES, EXTRACT APPENDED VALUES AND CHECK FOR -1
            left_shoulder_list.append(self.video_joint_dict["left_shoulder"][frame_num][2])
            right_shoulder_list.append(self.video_joint_dict["right_shoulder"][frame_num][2])
            left_ankle_list.append((self.video_joint_dict["left_heel"][frame_num][2] + self.video_joint_dict["left_foot_index"][frame_num][2]) / 2)
            right_ankle_list.append((self.video_joint_dict["right_heel"][frame_num][2] + self.video_joint_dict["right_foot_index"][frame_num][2]) / 2)

        #FOR VIDEO CAP
        """
        while True:
            ret, frame = cap.read()
            if frame is None or frame_count > self.stage_split:
                break
            pose_frame = np.copy(frame)
            self.pose_handler.findPose(pose_frame, draw = False)
            values = self.pose_handler.findPosition(pose_frame, ["left_shoulder", "right_shoulder", "left_heel", "left_foot_index", "right_heel", "right_foot_index"], draw=False)
            left_shoulder_list.append(values[0][2])
            right_shoulder_list.append(values[1][2])
            left_ankle_list.append((values[2][2] + values[4][2]) / 2)
            right_ankle_list.append((values[3][2] + values[5][2]) / 2)
            frame_count += 1
        """
        #AVG AROUND THE INITIAL FRAME
        self.hal = sum(left_ankle_list) / len(left_ankle_list)
        self.har = sum(right_ankle_list) / len(right_ankle_list)
        self.hsl = self.hal - (sum(left_shoulder_list) / len(left_shoulder_list))
        self.hsr = self.har - (sum(right_shoulder_list) / len(right_shoulder_list))

    def get_reference_jump_vals(self):
        #THIS ONE
        self.shoulder_levels = []
        left_shoulder_list = []
        right_shoulder_list = []
        left_ankle_list = []
        right_ankle_list = []

        for frame_num in range(0, self.frame_count):
            #TO ACCOUNT FOR MISSING VALUES, EXTRACT APPENDED VALUES AND CHECK FOR -1
            left_shoulder_list.append(self.video_joint_dict["left_shoulder"][frame_num][2])
            right_shoulder_list.append(self.video_joint_dict["right_shoulder"][frame_num][2])
            left_ankle_list.append((self.video_joint_dict["left_heel"][frame_num][2] + self.video_joint_dict["left_foot_index"][frame_num][2]) / 2)
            right_ankle_list.append((self.video_joint_dict["right_heel"][frame_num][2] + self.video_joint_dict["right_foot_index"][frame_num][2]) / 2)            

        #Finding the max shoulder values, and having ankle values reflect that
        #Can put an assert() clause here to ensure the same height
        max_shoulder_height = 2000
        for i in range(0, self.frame_count):
            left_shoulder_height = left_shoulder_list[i]
            right_shoulder_height = right_shoulder_list[i]
            avg_height = (left_shoulder_height + right_shoulder_height) / 2
            self.shoulder_levels.append(avg_height)
            if avg_height < max_shoulder_height:
                self.jsl = self.hal - left_shoulder_height
                self.jsr = self.har - right_shoulder_height
                max_shoulder_height = avg_height

    def estimate_head_height(self):
        height_estimate = int(self.shoulder_head_ratio_estimate * (self.hsl + self.hsr) / 2)
        self.head_point = int((self.hal + self.har) / 2) - height_estimate
    
    def calibrate_head_height(self, offset):
        new_head_point = self.head_point + offset
        height_pixels = int((self.hal + self.har) / 2) - new_head_point
        self.pixels_per_inch = height_pixels / self.height_measured
        self.log.info(f"calibrate_head_height(self, offset) | Pixels Per Inch: {self.pixels_per_inch}")
        self.head_point = new_head_point

    def draw_demo_frame(self, frame, frame_num):
        frame_cpy = np.copy(frame)
        vert = 12
        #TO ACCOUNT FOR MISSING VALUES, EXTRACT APPENDED VALUES AND CHECK FOR -1
        values = (self.video_joint_dict["left_shoulder"][frame_num], self.video_joint_dict["right_shoulder"][frame_num])


        lsy = values[0][2]
        lsx = values[0][1]
        rsy = values[1][2]
        rsx = values[1][1]
        sy = int((rsy + lsy) / 2)
        sx = int((rsx + lsx) / 2)
        by = int(((self.hal - self.hsl) + (self.har - self.hsr)) / 2)
        vert = (by - sy) / self.pixels_per_inch

        frame_cpy = cv.line(frame_cpy, (0, int((self.hal + self.har) / 2)), (1919, int((self.hal + self.har) / 2)), color=(255, 0, 0), thickness=2)
        frame_cpy = cv.line(frame_cpy, (lsx,lsy), (rsx,rsy), color=(0, 0, 255), thickness=2)
        frame_cpy = cv.line(frame_cpy, (sx, by), (sx, sy), color=(0, 0, 255), thickness=2)
        frame_cpy = cv.line(frame_cpy, (0, by), (1919, by), color=(255, 0, 0), thickness=2)
        return vert, frame_cpy


    def convert_to_formatted_frame(self, frame):
        #Going to take the data frame from opencv and converts it into numpy array
        #Size: (1080,1920,3)
        #3rd dimension is the colour in format B[0] G[1] R[2]
        self.log.info(f"Original Frame Shape: {frame.shape}")
        new_frame = np.flip(frame, 2)
        return new_frame

    def get_init_head_frame(self):
        return self.get_adjusted_head_frame(0)

    def get_adjusted_head_frame(self, offset):
        frame = np.copy(self.base_frame)
        image = cv.line(frame, (0, self.head_point + offset), (1919, self.head_point + offset), color=(0, 0, 255), thickness=1)
        image = cv.line(frame, (0, int((self.hal + self.har) / 2)), (1919, int((self.hal + self.har) / 2)), color=(255, 0, 0), thickness=1)
        ret_frame = self.convert_to_formatted_frame(image)
        return ret_frame

    def get_base_frame(self):
        ret_frame = self.convert_to_formatted_frame(self.base_frame)
        return ret_frame

    def play_selected_video():
        cap = cv.VideoCapture(f"{self.source_name}")
        while True:
            ret, frame = cap.read()
            cv.imshow("Sample Window", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(1/60)
            count+=1
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    print("Handlers for Measurement")
    ch = CalibrationHandler(source_name="vid_src\\ben_2.mp4")
    ch.generate_video_points()
    #ch.define_stages()