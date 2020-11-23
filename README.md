# Instructions
1. Clone the repository
2. Create a directory `model/` within local repository
3. Download [yolov4.onnx](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4/model) and move it to `model/`
4. Run `pip install -r requirements.txt` to get the dependencies
5. Your project is now set up

## Yolov3
1. download the model at from [here](https://drive.google.com/file/d/1wdW76M0VDxycQ81MSQWxxh08ermgD8te/view?usp=sharing) and unzip it.
2. make sure you have python 3.8 or later
3. `cd` into the directory and instal the requirements using `pip install -r requirements.txt`
4. run the model on an image using `python detect.py --source {IMG_PATH} --conf-thres {DETECTION_THRESHOLD}` where {IMG_PATH} is the path to the image and {DETECTION_THRESHOLD} is a float between 0.0 and 1.0

***
# Computer Vision For Crowd Detection

Detection and monitoring of crowds using computer vision has applications in crowd management and surveillance. Crowd management is important for public safety, especially now amidst the COVID-19 pandemic. Computer vision algorithms can assist with social distancing efforts aimed at slowing the spread of the virus, and alert when violations on the permitted headcount within a space occur. Crowd counting and localization can also be useful when designing public spaces such as airports and malls, and in making decisions on how to manage crowds in these public spaces.

***

###  Team Members
 - Tasneem Naheyan
 - Kenan Li
 - Inder Dhillon
 - Jing Li
 - Sadman Sakib
 - Reza Karimi
 - Youssef Guirguis

### Data
We will use the University of Amsterdam Multi-Camera Multi-Person dataset, which contains video clips of individuals and crowds interacting in two different environments. The dataset is labelled with ground truth pedestrian locations, and contains details on camera calibration. In case we run into problems with this dataset, we have identified the ‘EPFL Multi-camera Pedestrian dataset’ and ‘CityUHK-X: crowd dataset with extrinsic camera parameters’ as possible alternatives.
