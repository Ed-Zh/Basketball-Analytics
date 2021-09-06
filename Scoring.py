import numpy as np
from numpy.core.fromnumeric import mean 
import pandas as pd
from constants import *


def score(ts):
    '''
    Evaluate the closeness of the activation timings stored in ts. Assign a score 
    '''

    avg = mean(ts)
    score = 100 - 30 * sum([abs(avg-t) for t in ts])
    return score


def suggestions(d, sensitivity = 0.1):
    '''
    Analyze the peak-activation timings of muscle groups relative to the glutes (the most stable muscle when shooting)
    Give qualitative instructions on how to improve shot mechanics 
    '''
    glute_activation_time = d[HIP_RIGHT]
    analysis = []

    need_improvement = False
    for joint in JOINTS_ALL:
        activation_time = d[joint]

        if activation_time - glute_activation_time >= sensitivity:
            instruction = 'contracting your ' + joint_to_muscle[joint].lower() + ' earlier'
            need_improvement = True
        elif activation_time - glute_activation_time < -sensitivity:
            instruction = 'contract your ' + joint_to_muscle[joint].lower() + ' later'
            need_improvement = True
        else:
            instruction = 'maintaining the timing of your ' + joint_to_muscle[joint].lower() + ' activation'
            
        analysis.append(instruction)

    if not need_improvement:
        text = 'You have a very mechanically efficient shooting form.'
    else:
        text = 'To improve the mechanical efficiency of your shot, consider: \n'
        for instruction in analysis:
            text += instruction
            text += '\n'
    return text






