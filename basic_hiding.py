import picar_4wd as fc
import time
import numpy as np
from picamera2 import Picamera2
import cv2

speed = 10

ANGLE_RANGE = 180
STEP = 18
us_step = STEP
angle_distance = [0,0]
current_angle = 0
max_angle = ANGLE_RANGE/2
min_angle = -ANGLE_RANGE/2
scan_list = []
def scan_step(ref1, ref2):
        global scan_list, current_angle, us_step
        current_angle += us_step
        if current_angle >= max_angle:
                current_angle = max_angle
                us_step = -STEP
        elif current_angle <= min_angle:
                current_angle = min_angle
                us_step = STEP
        status = fc.get_status_at(current_angle, ref1=ref1, ref2=ref2)
        scan_list.append(status)
        if current_angle == min_angle or current_angle == max_angle:
                if us_step < 0:
                        scan_list.reverse()
                tmp = scan_list.copy()
                scan_list = []
                return tmp
        else:
                return False

def check_for_corner(scanList):
        # If front 2 sensor values are 0 (close to robot)
        if not any(scanList[4:6]):
                # If 2 right most sensor values are 0 (close to robot)
                if not any(scanList[8:]):
                        return "hardLeft"
                else:
                       return "hardRight"
        
        # If there is an object to the left of the robot
        elif any(np.array(scanList)[:5] == 0):
                return "slightRight"
        
        # If there is an object to the right of the robot
        elif any(np.array(scanList)[5:] == 0):
                return "slightLeft"
        
        else:
                return False


def check_for_dead_end(scanList):
        # Check if robot has reached a dead end

        ldist = int(np.round(np.mean(scanList[0:4])))
        cdist = int(np.round(np.mean(scanList[4:7])))
        rdist = int(np.round(np.mean(scanList[7:9])))
        distList = [ldist, cdist, rdist]
        if distList == [0,0,0] or distList == [1,0,1] or distList == [0,0,1] or distList == [1,0,0]:
                return True
        else:
                return False
                
        
        
def ninety_deg_turn(direction):
        # Command for robot to perform (near) 90 degree turn
        if direction == "left":
                fc.turn_left(speed)
                time.sleep(10/speed)
                fc.stop()
        elif direction == "right":
                fc.turn_right(speed)
                time.sleep(10/speed)
                fc.stop()


def adjust(direction, amt = 1):
        # Command for robot to perform slight adjustment based on amt parameter
        if direction == "left":
                fc.turn_left(speed)
                time.sleep(amt/speed)
                fc.forward(speed)
        elif direction == "right":
                fc.turn_right(speed)
                time.sleep(amt/speed)
                fc.forward(speed)

def main():
        cam = Picamera2()
        cam.preview_configuration.main.size = (320,240)
        cam.preview_configuration.main.format = "RGB888"
        
        cam.preview_configuration.align()
        cam.configure("preview")
        cam.start()

        # Adjust camera to not adjust to light to keep light readings consistent
        cam.set_controls({"AeEnable":False, 'AwbEnable': False, 'AnalogueGain': 3})
        
        # Face detection parameters
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"
        face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        face_detected = False
        face_det_time = 3       # Amount of time that face needs to stay detected
        face_det_start_time = time.time()
        stopwatch = 0

        
        first_iteration = True
        
        mode = "wait"
        turn_direction = None
        darkest_spot = None
        initial_hidden_threshold = 20
        hidden_threshold = initial_hidden_threshold

        # recording parameters
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter('output.mp4', fourcc, 25, (320,240)) 


        while True:
                scan_list = []
                img = cam.capture_array()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                
                print(mode)
                if mode == 'wait' or mode == 'ready':
                        # Get the height and width of the frame
                        (h, w) = img.shape[:2]
                        
                        # Preprocess the frame
                        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                                        (300, 300), (104.0, 177.0, 123.0))
                        
                        # Pass the blob through the network and obtain the detections and predictions
                        face_net.setInput(blob)
                        detections = face_net.forward()

                        face_detected = False
                        for i in range(0, detections.shape[2]):
                                confidence = detections[0, 0, i, 2]
                                
                                # Filter out weak detections
                                if confidence > 0.5:
                                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                        (startX, startY, endX, endY) = box.astype("int")
                                        
                                        # Draw the bounding box
                                        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)
                                        face_detected = True

                        stopwatch = time.time()-face_det_start_time        
                        if not face_detected and mode != 'ready':
                                face_det_start_time = time.time()
                        elif stopwatch > face_det_time:
                                mode = 'ready'


                                if not face_detected:
                                        mode = 'search'
                                        hidden_threshold = max(hidden_threshold, np.mean(gray))
                                        ninety_deg_turn('right')
                                        ninety_deg_turn('right')
                                        img = cam.capture_array()
                                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                        hidden_threshold = min(hidden_threshold, np.mean(gray))

                elif not mode == 'hidden':
                        start_hidden = time.time()
                hidden_timeframe = 1.5

                if mode == 'search' or mode == 'hide' or mode == 'hidden':

                        min_intensity, _, min_loc, _ = cv2.minMaxLoc(gray[int(gray.shape[1]*.5):, :])
                        
                        scan_list = scan_step(45, 25)
                        if not scan_list:
                                mode_print = mode + f" {stopwatch:.2f}" if mode == 'wait' else mode
                                img = cv2.putText(img, mode_print, (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                1, (255,0,0), 2, cv2.LINE_AA) 
                                out.write(img)  
                                cv2.imshow("view", img)
                                continue

                        if len(scan_list)< 10:
                                mode_print = mode + f" {stopwatch:.2f}" if mode == 'wait' else mode
                                img = cv2.putText(img, mode_print, (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                1, (255,0,0), 2, cv2.LINE_AA) 
                                out.write(img)  
                                cv2.imshow("view", img)
                                continue

                        turn_direction = check_for_corner(scan_list)

                        # Stop when in hiding spot
                        if np.sum(gray < hidden_threshold) >= gray.size*0.8:
                                mode = 'hidden'
                                print(mode)
                                if time.time() - start_hidden > hidden_timeframe or turn_direction in ["hardRight", "hardLeft"]:
                                        fc.stop()
                                        print("hidden succesfully.")
                                        break

                        if not first_iteration:
                                deadEnd = check_for_dead_end(scan_list)
                                
                                if turn_direction == "hardRight":
                                        ninety_deg_turn("right")
                                elif turn_direction == "slightRight":
                                        adjust("right", 3)
                                elif turn_direction == "hardLeft":
                                        ninety_deg_turn("left")
                                elif turn_direction == "slightLeft":
                                        adjust("left", 3)
                                elif deadEnd:
                                        fc.backward(speed)
                                        ninety_deg_turn("right")
                                        ninety_deg_turn("right")
                                
                                elif min_intensity < hidden_threshold:
                                        mode = 'hide'
                                        print(darkest_spot)

                                        if darkest_spot != 'center':
                                                if min_loc[0] > gray.shape[1]* (7/9):
                                                        darkest_spot = 'right'
                                                        adjust('right')
                                                elif min_loc[0] < gray.shape[1] * (3/9):
                                                        darkest_spot = 'left'
                                                        adjust('left')
                                                else:
                                                        darkest_spot = 'center'
                                        else:
                                                fc.forward(speed)
                                else:
                                        mode = 'search'
                                        fc.forward(speed)
                        else:
                                if hidden_threshold == initial_hidden_threshold:        # If face detection skipped
                                        hidden_threshold = max(hidden_threshold, np.mean(gray))
                        first_iteration = False

                mode_print = mode + f" {stopwatch:.2f}" if mode == 'wait' else mode
                img = cv2.putText(img, mode_print, (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255,0,0), 2, cv2.LINE_AA) 
                out.write(img)  
                cv2.imshow("view", img)
                
                k = cv2.waitKey(30) & 0xff
                if k == 27: # press 'ESC' to quit
                        break

        cam.close()


if __name__ == "__main__":
        try: 
                main()
        finally: 
                fc.stop()
