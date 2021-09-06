import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import cv2
import mediapipe as mp

from constants import *


class Analyzer():



    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        
        # Parameters analyzed from the video.
        self.all_positions = pd.DataFrame()
        self.angular_acceleration = {} # Each joint (3-tuple) should correspond to a list of angular accelerations by time
        self.avg_alpha = {} # Each joint (3-tuple) should correspond to its average angular acceleration
        self.joint_colors = {} # Each joint (3-tuple) should correspond to a list of color labels by time
        self.acceleration = np.array([])
        self.avg_a = 0
        self.speed = np.array([])
        self.avg_v = 0
        self.size = 1
        self.strength = 1

        self.width = 0
        self.height = 0
        self.path = ''
        self.framerate = 30
        
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
    
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
       
        static_image_mode = False
        upper_body_only = False
        smooth_landmarks = True
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5

    # Elementary functions for processing the video 


    def findPose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=False):
        lmList = []
        assert self.results.pose_landmarks
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


    def find_all_positions(self, cap):
        positions = pd.DataFrame(columns=range(33))
        while True:

            success, img = cap.read()

            if not success:
                self.all_positions = positions
                break

            img = self.findPose(img)
            raw_position = self.findPosition(img, draw=False)
            clean_position = [(pos[1], pos[2])for pos in raw_position]
            row = {i: clean_position[i] for i in range(33)}
            positions = positions.append(row, ignore_index=True)

        return
    


    # Functions for analyzing the body mechanics


    def angle_horizontal(self, joints):
        # Return the angle of a limb w.r.t the x-axis
        a, b = joints[0], joints[1]
        joint_a = self.all_positions[a].apply(lambda x: np.array(x))
        joint_b = self.all_positions[b].apply(lambda x: np.array(x))
        diff = joint_a - joint_b
        diff = np.concatenate(diff).reshape(-1,2)

        return savgol_filter(np.arctan2(-diff[:,1],diff[:,0]) * 180 / np.pi,5,2)


    def angle_between(self,joint):
        assert len(joint) == 3
        (left, mid, right) = joint

        '''
        # Version 1
        angle_1 = self.angle_horizontal((mid,left))
        angle_2 = self.angle_horizontal((mid,right))
        return savgol_filter(np.abs(angle_1 - angle_2),7,2)

        '''
        
        joint_left = np.concatenate(self.all_positions[left].apply(lambda x: np.array(x))).reshape(-1,2)
        joint_mid = np.concatenate(self.all_positions[mid].apply(lambda x: np.array(x))).reshape(-1,2)
        joint_right = np.concatenate(self.all_positions[right].apply(lambda x: np.array(x))).reshape(-1,2)

        l = joint_left - joint_mid
        r = joint_mid - joint_right

        lsize = np.power(l[:,0] ** 2 + l[:,1] ** 2, 0.5)
        rsize = np.power(r[:,0] ** 2 + r[:,1] ** 2, 0.5)

        omega = (np.pi - np.arccos(np.sum(l * r, axis = 1)/(lsize*rsize))) * 180 / np.pi
        return savgol_filter(omega,7,2)









    def compute_angular_acceleration(self,joints):
        angles = self.angle_between(joints)
        acc = np.gradient(angles,2)
        self.angular_acceleration[joints] = acc
        return savgol_filter(acc,5,2)


    def compute_v_a(self,joint):
        positions = np.concatenate(self.all_positions[joint].apply(lambda x: np.array(x))).reshape(-1,2)
        x = savgol_filter(positions[:,0],7,2)
        y = savgol_filter(positions[:,1],7,2)


        # Speed
        vx = np.gradient(x)
        vy = np.gradient(y)
        speed = np.power(vx ** 2 + vy ** 2,0.5) 
        self.speed = speed
        self.avg_v = (speed * (speed > 0)).mean()
        # Acceleration
        
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        acc = np.power(ax ** 2 + ay ** 2,0.5) 
        '''
        acc = np.gradient(speed)
        '''
        self.acceleration = acc
        self.avg_a = (acc * (acc > 0)).mean()
        return self.avg_a, self.acceleration


    def estimate_body_size(self):
        shoulder = np.concatenate(self.all_positions[12].apply(lambda x: np.array(x))).reshape(-1,2)
        hip = np.concatenate(self.all_positions[24].apply(lambda x: np.array(x))).reshape(-1,2)
        knee = np.concatenate(self.all_positions[26].apply(lambda x: np.array(x))).reshape(-1,2)


        shoulder = np.concatenate([savgol_filter(shoulder[:,0],7,2),savgol_filter(shoulder[:,1],7,2)]).reshape(2,-1).T
        hip = np.concatenate([savgol_filter(hip[:,0],7,2),savgol_filter(hip[:,1],7,2)]).reshape(2,-1).T
        knee = np.concatenate([savgol_filter(knee[:,0],7,2),savgol_filter(knee[:,1],7,2)]).reshape(2,-1).T

        d1 = shoulder - hip
        d2 = hip - knee
        length_total = np.power(d1[:,0]**2 + d1[:,1]**2 ,0.5) + np.power(d2[:,0]**2 + d2[:,1]**2 ,0.5)
        self.size = length_total.mean()

        return self.size
    

    def set_strength(self, strength = 1):
        self.strength = strength
        return


    def scale(self,use_size=True, use_strength = False, use_framerate = True):
        factor = 1

        if use_size:
            factor /= self.size
        if use_strength:
            factor /= self.strength
        if use_framerate:
            factor *= self.framerate 
        
        return factor


        # Functions for drawing 


    def connect_joints(self,img,joints,t,color):
        (a, b) = joints
        a_coordinates = self.all_positions.iloc[t][a]
        b_coordinates = self.all_positions.iloc[t][b]
        #print(a_coordinates)
        return cv2.line(img,a_coordinates,b_coordinates,color=color,thickness=10)


    def colorize(self, a, low = 1, high = 3):
        # helper function for the following function 
        if a < low:
            return (0,255,0)
        elif a < high:
            s = (a-low)/(high-low)
            return (0, int((1-s)*255),int(s*255)) 
        else:
            return (0,0,255)


    def colorize_angular_acc(self, alpha, low = 1, high = 3):
        alpha = alpha * (alpha > 0)
        avg = alpha.mean()
        ratios = alpha/avg
        return avg, [self.colorize(ratio, low, high) for ratio in ratios]


        # High level APIs wrapping the functions above


    def analyze(self,path,list_of_joints):
        # Wrap all preliminary analysis 
        self.path = path
        cap = cv2.VideoCapture(path)

        self.width = int(cap.get(3))
        self.height = int(cap.get(4))
        self.framerate = cap.get(cv2.CAP_PROP_FPS)


        self.find_all_positions(cap)

        self.joints_analyzed = list_of_joints

        low, high = 1.0, 3.0

        for joints in list_of_joints:
            acc = self.compute_angular_acceleration(joints)
            self.angular_acceleration[joints] = acc

            avg, labels = self.colorize_angular_acc(acc,low,high)
            self.avg_alpha[joints] = avg
            self.joint_colors[joints] = labels
                
        self.compute_v_a(16) # Right Wrist

    def score_motion(self):
        '''
        Evaluate the closeness of the activation timings stored in ts. Assign a score 
        '''

        import Scoring
        timings = [np.argmax(alpha)/len(alpha) for alpha in self.angular_acceleration.values()] 
        score = Scoring.score(timings)

        return score


    def give_suggestions(self):
        '''
        Analyze the peak-activation timings of muscle groups relative to the glutes (the most stable muscle when shooting)
        Give qualitative instructions on how to improve shot mechanics
        '''

        import Scoring

        d = {joint:np.argmax(alpha)/len(alpha) for joint,alpha in self.angular_acceleration.items()}
        return Scoring.suggestions(d,0.1)





    def output_video(self,name = 'output', limbs = [LEG_LOWER_RIGHT, LEG_UPPER_RIGHT, UPPER_BODY_RIGHT], out_frame_rate = 12):

        cap = cv2.VideoCapture(self.path)
        t = 0 # counting frames

            
        outpath = 'Videos/' + name + '.avi'
        out = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('M','J','P','G'), out_frame_rate, (self.width,self.height))

        while True:
            success, img = cap.read()
            if not success:
                break
            limb_color = (255,0,0)
            
            for limb in limbs:
                img = self.connect_joints(img,limb,t,limb_color)

            for i, joint in enumerate(self.joints_analyzed):
                # Radius of the circle at each joint
                intensity = max(0,self.angular_acceleration[joint][t] / self.avg_alpha[joint])
                r = 5 + int(6*np.sqrt(intensity))
                color = self.joint_colors[joint][t]
                center = self.all_positions.iloc[t][joint[1]]
                img = cv2.circle(img,center=center,radius=r,color=color,thickness=cv2.FILLED) # Draw the circle around the joint


                shift_horizontal_text = i * 60
                img = cv2.putText(img,joint_to_text[joint], org=(int(0.1*self.width) + shift_horizontal_text, int(0.9*self.height)),fontFace= cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255), thickness=2,lineType=cv2.LINE_AA)
                max_length = int(0.1 * self.height)
                shift_horizontal_bar = 25
                shift_vertical = -40
                length = 10 + int(min(max_length, intensity*max_length * 1/3))
                bot = (int(0.1*self.width) + shift_horizontal_bar + shift_horizontal_text, int(0.9*self.height) + shift_vertical)
                top = (int(0.1*self.width) + shift_horizontal_bar + shift_horizontal_text, int(0.9*self.height) + shift_vertical - length)
                img = cv2.line(img,bot,top,color=color,thickness=25, lineType=cv2.LINE_8)

            # Tracking right wrist
            img = cv2.putText(img,'Overall(wrist)', org=(int(0.1*self.width), int(0.9*self.height) + 50),fontFace= cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255), thickness=2,lineType=cv2.LINE_AA) # Text
            max_length = int(0.2 * self.height)
            intensity = max(0,self.acceleration[t] / self.avg_a)
            color = self.colorize(intensity)
            length = 10 + int(min(max_length, intensity*max_length * 1/3))
            left = (int(0.1*self.width) + 25, int(0.9*self.height) + 80)
            right = (int(0.05*self.width) + 25 + length, int(0.9*self.height) + 80)
            img = cv2.line(img,left,right,color=color,thickness=20, lineType=cv2.LINE_8)

            # Draw a square around the wrist
            (x,y) = self.all_positions.iloc[t][16]
            size = 20
            tl = (int(x - size/2), int(y - size/2))
            br = (int(x + size/2), int(y + size/2))
            img = cv2.rectangle(img,tl,br,color,cv2.FILLED)

            #cv2.imshow('Image', img)   
            #cv2.waitKey(1)
            out.write(img)
            t += 1
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return

    def output_graph(self,name = 'mechanics analysis', cols_to_analyze = "All", scaling = {'size': True, 'strength': False, 'framerate': True }):
        alpha = pd.DataFrame(self.angular_acceleration)
        alpha.columns = [joint_to_text[joint] for joint in self.angular_acceleration.keys()]
        alpha = alpha.apply(lambda a: a/max(a), axis = 0)
        alpha = alpha.apply(lambda a: savgol_filter(a*(a>0),3,1))
        t = np.linspace(0,1,len(alpha))

        if cols_to_analyze == "All":
            cols = alpha.columns
        else:
            cols = [col for col in cols_to_analyze if col in alpha.columns]


        fig, (ax1, ax2) = plt.subplots(2,figsize = (16,12))
        fig.suptitle('Muscle Activation and Ball Acceleration')

        ax1.plot(t,alpha[cols])
        ax1.set(xlabel="relative time", ylabel='relative activation')
        ax1.legend(cols)
        
        scale = self.scale(use_size=scaling['size'], use_strength=scaling['strength'], use_framerate=scaling['framerate'])
        a = savgol_filter(self.acceleration * scale / 50,7,3)
        v = savgol_filter(self.speed * scale / 150,7,3)


        ax2.plot(t,v)
        ax2.plot(t,a)
        ax2.set(xlabel="relative time", ylabel='relative speed/acceleration')
        ax2.legend(['speed','acceleration'])

        fig.text(0.2, 0.47, 'Overall Score: ' + str(self.score_motion()), horizontalalignment='left',verticalalignment='center',fontsize = 15, family = 'sans-serif', color = 'red')
        fig.text(0.2, 0.4,self.give_suggestions(),horizontalalignment='left',verticalalignment='center',fontsize = 12, family = 'sans-serif',color = 'blue')
        fig.savefig(('Graphs/' + name + '.pdf'))

        return fig




            

def pipeline(path,output_name = 'analysis', joints = [KNEE_RIGHT,HIP_RIGHT], limbs = [LEG_LOWER_RIGHT, LEG_UPPER_RIGHT, UPPER_BODY_RIGHT],out_frame_rate = 12):
    detector = Analyzer()
    detector.analyze(path,joints)
    detector.output_video(name = output_name, limbs = limbs, out_frame_rate = out_frame_rate)
    detector.output_graph(name = output_name)
    
'''

def main():
    detector = poseDetector()
    name = 'Ray Shooting'
    format = '.MOV'
    path = 'Videos/' + name + format
    list_of_joints = [KNEE_RIGHT, HIP_RIGHT]
    detector.analyze(path, list_of_joints)

    cap = cv2.VideoCapture(path)
    t = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    outpath = 'Videos/' +name + '.avi'
    out = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('M','J','P','G'), 12, (frame_width,frame_height))


    while True:
        success, img = cap.read()
        if not success:
            break
        limb_color = (255,0,0)
        img = detector.connect_joints(img,LEG_LOWER_RIGHT,t,limb_color)
        img = detector.connect_joints(img,LEG_UPPER_RIGHT,t,limb_color)
        img = detector.connect_joints(img,UPPER_BODY_RIGHT,t,limb_color)

        for joints in list_of_joints:
            r = 10 + max(0,int(5 * (detector.angular_acceleration[joints][t] / detector.avg_alpha[joints])))
            color = detector.joint_colors[joints][t]
            center = detector.all_positions.iloc[t][joints[1]]
            img = cv2.circle(img,center=center,radius=r,color=color,thickness=cv2.FILLED)

        cv2.imshow('Image', img)


        #print(lmList)
        #img = cv2.line(img,(0,0),(300,500),(255,0,0))
        # print(lmList)
        #cTime = time.time()
        #fps = 1/(cTime-pTime)
        #img = cv2.putText(img, str(int(fps)), org=(300, 500), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(100, 100, 100), thickness=3)
        cv2.waitKey(1)
        out.write(img)

        t += 1
    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''