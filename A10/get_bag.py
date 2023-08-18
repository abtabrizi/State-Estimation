#! /usr/bin/python3

# Author: Arash
# Node for extracting images from a rosbag in the format needed for ORB-SLAM3.
# Referenced https://answers.ros.org/question/283724/saving-images-with-image_saver-with-timestamp/
# Code inspired by https://github.com/kevin-robb/orb_slam_implementation/blob/main/scripts/create_timestamps.py

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess
import sys
import os

bridge = CvBridge()
# folder name for dataset
CAM0_PATH = None
CAM1_PATH = None
# Ensure images have time correspondance.
timestamps = []
cam1_index = -1


def get_cam0(msg):
    global timestamps
    # cam0 will control the time.
    time = msg.header.stamp
    timestamps.append(time)
    # write the next timestamp to the file.
    timestamps_file.write(str(time)+"\n")
    try:
        # Convert the Image msg to OpenCV2 object.
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # Save image with time as name.
        if not cv2.imwrite(os.path.join(CAM0_PATH, str(time)+'.png'), cv2_img):
            raise Exception("Failed to write image.")
    except:
        rospy.logerr("Exception encountered on cam0.")


def get_cam1(msg):
    global cam1_index
    # use the time from cam0.
    cam1_index += 1
    # wait if necessary until cam0 has come in for this timestep.
    time = None
    while len(timestamps)-1 < cam1_index:
        rospy.sleep(0.05)
    try:
        time = timestamps[cam1_index]
    except:
        rospy.logerr("cam1 unable to sync with cam0 at timestamp "+str(len(timestamps))+", index "+str(cam1_index))
    
    try:
        # Convert the Image msg to OpenCV2 object.
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # Save image with time as name.
        if not cv2.imwrite(os.path.join(CAM1_PATH, str(time)+'.png'), cv2_img):
            raise Exception("Failed to write image.")
    except:
        rospy.logerr("Exception encountered on cam1.")


def run_bash_cmd(command:str):
    # run something on the command line.
    process = subprocess.Popen(command.split())
    output, error = process.communicate()


def main():
    global timestamps_file, CAM0_PATH, CAM1_PATH
    rospy.init_node('image_conversion_node')

    DATASET_NAME = 'traj3_files'
    
    # create the folder that images will be saved to.
    DIR_PATH = os.path.join(os.path.expanduser('~/Datasets/traj3/'), DATASET_NAME)
    CAM0_PATH = os.path.join(DIR_PATH, 'images', 'mav0', 'cam0', 'data')
    CAM1_PATH = os.path.join(DIR_PATH, 'images', 'mav0', 'cam0', 'data')
    run_bash_cmd("mkdir -p "+DIR_PATH+"/images/mav0/cam0/data")
    run_bash_cmd("mkdir -p "+DIR_PATH+"/images/mav0/cam1/data")
    rospy.loginfo("Images will be saved to "+DIR_PATH+".\nYou can now run 'rosbag play ROSBAG_NAME.bag' in a new terminal.\nClose this node with Ctrl+C when your rosbag has finished playing.")
    # Init file for timestamps.
    timestamps_file = open(DIR_PATH+"/timestamps.txt", "w")
    
    # Subscribe to the image streams.
    rospy.Subscriber('/B05/D435i/color/image_raw', Image, get_cam0)
    rospy.Subscriber('/B05/D435i/depth/image_rect_raw', Image, get_cam1)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
