import mediapipe as mp
import cv2 as cv
import time
import numpy as np
import mapping

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
    def __init__(self):
        print("Calibration Handler Created")
        #self.source_name = "Down_And_Up_2.mp4"
        #self.source_name = "No_Strength_Shortening_1.mp4"
        self.source_name = "ben_2.mov"
        self.base_frame = None
        self.stage_tolerance = 4 #CONFIGURATION VALUE
        #self.height_measured = 77.5 #CONFIGURATION VALUE
        self.height_measured = 72.75
        self.num_max_vals = 1 #CONFIGURATION VALUE
        self.shoulder_head_ratio_estimate = 1.25 #INIT - CHANGES

        self.head_point = 500
        self.pose_handler = PoseHandler()
        self.stage_split = 0
        self.pixels_per_inch = 5

    def setup_demo(self):
        self.demo_vidcap = cv.VideoCapture(f"vid_src\\{self.source_name}")

    def get_demo_frame(self):
        ret, frame = self.demo_vidcap.read()
        if frame is None:
            self.demo_vidcap.release()
            self.demo_vidcap = cv.VideoCapture(f"vid_src\\{self.source_name}")
            ret, frame = self.demo_vidcap.read()
        frame = self.draw_demo_frame(frame=frame)
        ret_frame = self.convert_to_formatted_frame(frame=frame)
        return ret_frame

    def close_demo(self):
        self.demo_vidcap.release()

    def define_stages(self):
        split_found = 0
        count = 0
        #Code for displaying a video for 60fps
        cap = cv.VideoCapture(f"vid_src\\{self.source_name}")
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
        self.get_stage_1_vals()
        print(self.hsr, self.jsr)
        pixel_vertical = int(((self.jsl-self.hsl) + (self.jsr-self.hsr))/2)
        inch_vertical = pixel_vertical / self.pixels_per_inch
        print(f"Vertical in Pixels: {pixel_vertical} | Vertical in Inches: {inch_vertical}")
        return inch_vertical

    def get_stage_0_vals(self):
        frame_count = 0
        cap = cv.VideoCapture(f"vid_src\\{self.source_name}")
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

    def get_stage_1_vals(self):
        frame_count = 0
        cap = cv.VideoCapture(f"vid_src\\{self.source_name}")
        left_shoulder_list = []
        right_shoulder_list = []
        left_ankle_list = []
        right_ankle_list = []
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            if frame_count < self.stage_split:
                frame_count += 1
                continue
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
        self.pixels_per_inch = int(height_pixels / self.height_measured)
        print(f"Pixels Per Inch: {self.pixels_per_inch}")
        self.head_point = new_head_point

    def draw_demo_frame(self, frame):
        frame_cpy = np.copy(frame)
        vert_jump = 12
        #NEED:
        # - Text showing the current jump height
        # - Base lines for the ankles and the shoulders (HAL/HAR/HA)
        # - Line for the current shoulders
        # - Vertical Line Connecting the two shoulder lines
        frame_cpy = cv.putText(frame_cpy, f"Current Jump Height: {vert_jump}", (0, 0), 3, 4, (255, 255, 255, 255), 2)
        frame_cpy = cv.line(frame_cpy, (0, int((self.hal + self.har) / 2)), (1919, int((self.hal + self.har) / 2)), color=(255, 0, 0), thickness=1)
        frame_cpy = cv.line(frame_cpy, (0, int(((self.hal - self.hsl) + (self.har - self.hsr)) / 2)), (1919, int(((self.hal - self.hsl) + (self.har - self.hsr)) / 2)), color=(255, 0, 0), thickness=1)
        return frame_cpy


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
        cap = cv.VideoCapture(f"vid_src\\{self.source_name}")
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