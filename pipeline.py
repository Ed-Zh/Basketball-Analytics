import cv2
import AnalyzerModule as pm
import numpy as np
import mediapipe as mp

import matplotlib.pyplot as plt

joints = [pm.SHOULDER_RIGHT,pm.HIP_RIGHT,pm.KNEE_RIGHT,pm.ANKLE_RIGHT,pm.ELBOW_RIGHT]
limbs = [pm.ARM_LOWER_RIGHT,pm.ARM_UPPER_RIGHT,pm.UPPER_BODY_RIGHT,pm.LEG_UPPER_RIGHT,pm.LEG_LOWER_RIGHT, pm.FOOT_RIGHT]

def analyze_multiple_players(names):
    for name in names:
        path = 'Videos/' + name + '.MOV'
        pm.pipeline(path = path, output_name = name, joints=joints,limbs=limbs, out_frame_rate=12)

players1 = ['Ray', 'Jeremy', 'Trae']
players2 = ['Edward 1', 'Edward 2']
analyze_multiple_players(players2)

