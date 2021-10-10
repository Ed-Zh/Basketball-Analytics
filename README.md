# Basketball-Analytics
Investigated quantitatively the optimal body mechanics for shooting basketball by employing pose estimation and sports theory.



## Background 

If there is one thing that I wish I had learned earlier in my ten years of experience in basketball, it is shot mechanics. 



Too often, when training for better performance, beginners put too much emphasis on the drills: "I should complete 80 behind-the-back dribbles, 100 free throws, 50 three-pointers, etc." Focusing on the drills usually means very little attention paid to their body mechanics during the moves. How much should I bend my knees? Should I contract my glute first or quad first? 



As a wise man once observed, "form before strength," which means one should priortize body mechanics over brute force in most skill-intensive sports.  The attention to details and mind-muscle connection is often what sets professional players apart from street hoopers. But how can we learn from those superstars? What's the secret magic that helps them optimize their forms to achieve longer shooting range and higher explosiveness?



Though there are rule-of-thumbs about shooting forms, little quantitative work has been done regarding the underlying body mechanics. By employing pose estimation and sports theory, this project aims to unscramble to myth behind optimal shot mechanics in basketball. 



## Research Question

Different key muscle groups are activated at different time intervals when a player shoots. 

**What's the optimal timing and order for activating these muscle groups?** 



## Model Methodology

1. Use the *mediapipe* API to obtain the locations of each joint (e.g. ankle, knee, hip)
2. Connect neighbouring joints to model limbs (e.g. connecting hip and knee gives the line representing thigh)
3. Compute the angular acceleration at each joint using the limbs stemming from it (e.g. the rotational movement thigh and lower leg determines the angular acceleration at knee)
4. Estimate the activation of a specific muscle group using angular acceleration at the corresponding joint (e.g. glute corresponds to hip). This follows from the angular part of Newton's Second Law (angular acceleration proportional to torque)

## Addtional Functionality

1. Produce a graph describing the player's shot mechanics as muslce activation vs. time
2. Evaluate the efficiency of the shot as a score 
3. Give suggestions on how to improve the shot mechanics

Note: the speed and acceleration are in relative terms, scaled for comparision across players



## Codes

The *AnalyzerModule.py* file wraps an *Analyzer* class, the instance of which can be used to analyze a video of a player shooting. 

The *pipeline.py* file applies *Analyzer* to produce i) a slow-motion video tracking (at real time) the activation of each muscle groups and ii) a graph illustrating the graph. 

The *Scoring.py* file implements a scoring system to evaluate the efficiency of the player's shot mechanics.



## Demo

See the Demo folder for demo videos and graphs.

