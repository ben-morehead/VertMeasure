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
    def __init__(self, source_name="ben_2.mov", jumper_name="Ben", jumper_height=72, jump_style=4, log=logging.getLogger(__name__)):
        print("Calibration Handler Created")
        self.source_name = source_name
        self.log = log
        #self.source_name = "No_Strength_Shortening_1.mp4"
        #self.source_name = "ben_2.mov"
        self.jumper_name= jumper_name
        self.jump_style = jump_style

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

    def setup_demo(self):
        self.demo_vidcap = cv.VideoCapture(f"{self.source_name}")

    def get_demo_frame(self):
        ret, frame = self.demo_vidcap.read()
        if frame is None:
            self.demo_vidcap.release()
            self.demo_vidcap = cv.VideoCapture(f"{self.source_name}")
            ret, frame = self.demo_vidcap.read()
        vert, frame = self.draw_demo_frame(frame=frame)
        ret_frame = self.convert_to_formatted_frame(frame=frame)
        return vert, ret_frame

    def close_demo(self):
        self.demo_vidcap.release()
        self.demo_vidcap = None

    def export_jump_info(self):
        header = ["Name", "Maximum Vertical Jump (inches)", "Descent Speed (inches/s)", "Descent Level (inches)", "Ground Time (s)"]
        if self.jump_style == 4:
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

            
        print("ADVANCED DATA INFO")
        print(f"Init Frame = {init_frame}")
        print(f"Init Level = {init_level}")
        print(f"Descent Frame = {descent_frame}")
        print(f"Descent Level = {lowest_point}\n")
        print(f"Final Ground Frame = {final_ground_frame}")

        descent_level = (lowest_point - init_level) / self.pixels_per_inch
        descent_speed = descent_level / ((descent_frame - init_frame) / self.base_fps)
        ground_time = (final_ground_frame - init_frame) / self.base_fps
        print(f"Lowest Reached (inches): {descent_level:.2f}")
        print(f"Descent Speed (inches/second): {descent_speed:.2f}")
        print(f"Ground Time: {ground_time:.2f} seconds")
        return descent_level, descent_speed, ground_time
        

    def define_stages(self, one_leg=False):
        split_found = 0
        count = 0
        #Code for displaying a video for 60fps
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

    def calculate_vertical_jump(self):
        if self.jump_style == 4:
            self.get_two_foot_vals()
        else:
            print("JUMPING STYLE NOT YET IMPLEMENTED")
        pixel_vertical = int(((self.jsl-self.hsl) + (self.jsr-self.hsr))/2)
        inch_vertical = pixel_vertical / self.pixels_per_inch
        print(f"Vertical in Pixels: {pixel_vertical} | Vertical in Inches: {inch_vertical}")
        self.inch_vert = inch_vertical
        return inch_vertical

    def get_stage_0_vals(self):
        frame_count = 0
        cap = cv.VideoCapture(f"{self.source_name}")
        left_shoulder_list = []
        right_shoulder_list = []
        left_ankle_list = []
        right_ankle_list = []
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

        self.hal = sum(left_ankle_list) / len(left_ankle_list)
        self.har = sum(right_ankle_list) / len(right_ankle_list)
        self.hsl = self.hal - (sum(left_shoulder_list) / len(left_shoulder_list))
        self.hsr = self.har - (sum(right_shoulder_list) / len(right_shoulder_list))

    def get_two_foot_vals(self):
        self.shoulder_levels = []
        frame_count = 0
        cap = cv.VideoCapture(f"{self.source_name}")
        left_shoulder_list = []
        right_shoulder_list = []
        left_ankle_list = []
        right_ankle_list = []
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            #if frame_count < self.stage_split:
                #frame_count += 1
                #continue
            pose_frame = np.copy(frame)
            self.pose_handler.findPose(pose_frame, draw = False)
            values = self.pose_handler.findPosition(pose_frame, ["left_shoulder", "right_shoulder", "left_heel", "left_foot_index", "right_heel", "right_foot_index"], draw=False)
            left_shoulder_list.append(values[0][2])
            right_shoulder_list.append(values[1][2])
            left_ankle_list.append((values[2][2] + values[4][2]) / 2)
            right_ankle_list.append((values[3][2] + values[5][2]) / 2)
            frame_count += 1

        #Finding the max shoulder values, and having ankle values reflect that
        #Can put an assert() clause here to ensure the same height
        max_shoulder_height = 2000
        for i in range(0, len(left_shoulder_list)):
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
        print(f"Pixels Per Inch: {self.pixels_per_inch}")
        self.head_point = new_head_point

    def draw_demo_frame(self, frame):
        frame_cpy = np.copy(frame)
        vert = 12
        #NEED:
        # - Text showing the current jump height
        # - Base lines for the ankles and the shoulders (HAL/HAR/HA)
        # - Line for the current shoulders
        # - Vertical Line Connecting the two shoulder lines
        self.pose_handler.findPose(frame_cpy, draw = False)
        values = self.pose_handler.findPosition(frame_cpy, ["left_shoulder", "right_shoulder"], draw=False)

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
    ch = CalibrationHandler()
    ch.define_stages()