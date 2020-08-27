# ICRA2020_RM_Perecption&Planning

## 1 Preface
Because of the 2020 epidemic, school management does not allow us to go back to school, so most of our work is online.

### 1.1 Hardware
We test the inference speeds of Deep-FSMs on an NVIDIA Jetson TX2, a GPU platform targets at power-constrained mobile applications.

## 2 Perecption
The location of our robot is obtained by lidar. First, the map information is established by the Gmapping algorithm. Second, robotic position and orientation are estimated by the adaptive Monte Carlo localization algorithm (AMCL) based on the distance and speed information obtained by lidar and odometer. In order to lower the complexity of the algorithm, we use KLD Sampling, which is a variant of AMCL using Kullback-Leibler divergence (KLD) for error estimate, to locate the robot in real-time. Precise localization of an autonomous robot is the basis of path planning.
the location of the enemy  mainly depends on two outpost cameras, which look down at the field from two meters high. We can see the image of a certain angle of the field through one of the outpost cameras, and they both supplement each other's visual blind area. Thus, we can get the global information of the field by analyzing the data from them. 

![](https://github.com/gongpx20069/ICRA_RM_Perception-Planning/blob/master/img/yolo.jpg)

First, we use YOLO nano to detect the enemy in the images of both cameras. After detecting the center position of the enemy's bounding box, we use binocular vision to calculate the spatial coordinates, and finally get the location of the enemy

![](https://github.com/gongpx20069/ICRA_RM_Perception-Planning/blob/master/img/global.jpg)

## 3 Planning

our robot uses global path planning based on A-star algorithm. After getting the global path, we use the timed elastic band algorithm to get the local path which is easier for the robot to execute. In the 2D simulation, our work is much simpler. In the part of path planning, we use A-star algorithm based on the probabilistic roadmap.
The path planning of 2D simulation is shown in the figure below.

![](https://github.com/gongpx20069/ICRA_RM_Perception-Planning/blob/master/img/prm.png)

Where, the blue dots and lines indicate the path that the robot can walk, and the red line represents the path of real-time planning. Some blue lines may pass through obstacles, and we will eliminate them in real time.

![](https://github.com/gongpx20069/ICRA_RM_Perception-Planning/blob/master/img/2D2v2.png)

## 4 Codes

### 4.1 YoloNano
#### 4.1.1 YoloNanoTrain.ipynb
This is our reproduction of Yolo nano's training script.
#### 4.1.2 YoloNanoTest.ipynb
This is our reproduction of Yolo nano's test script. We can also directly convert to Python in ipynb and modify the code to run in NVIDIA Jetson TX2.
### 4.2 sentry_pose
#### 4.2.1 sentry_train.py
We collected the position of our robot after AMCL algorithm and the information of two sentry cameras, stored the relevant data and carried out network training. Our network can directly input the sentry camera and output the position of the robot.

#### 4.2.2 sentry_test.py
Neural network input sentry image directly to predict the position of enemy robot.

### 4.3 path planning
#### 4.3.1 