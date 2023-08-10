# Project summary

This project aims to develop a tool to help patients in scoliosis treatment by physiotherapy and give clinicains a first quantification of this type of treatment, through symmetry indices. The main tool is a GUI (interface) including exercise sequence visualization, markers detection and correction, symmetry indices computation and evolution plots, depth maps and informations saving.

Main files contains everything to run the interface. Notebooks are supplementary files created, mostly work in progress or experiments included in main files when completed.

# Main files

interface.py contains the code to run the interface (main tool) and all its methods
design_interface.kv is the Kivy script to build the interface

interface_tapir.py is another version of the interface.py file, integrating the TAPIR model to track markers

To run the TAPIR model, you have to uncomment the TAPIR button section in desing_interface.kv (lines 113-117)

read_raw_file.py contains the code to convert raw data to intensity images and xyz data to arrays. It also includes the methods to retrieve xyz coordinates from markers positions in intensity images.
marker_detection.py contains the code to preprocess the intensity images and automatically detect the markers on all frames (after preprocessing)

# Notebooks

automatic_crop.ipynb gathers experiments to find the best cropping parameters and an algorithm that fits every participant and movement
depth_map.ipynb contains data processing to create depth maps with a great contrast and a customized colormap, has been integrated in interface.py
rotate_pc.ipynb contains functions and illustrations of body rotation to align the patient with the acquisition system reference, has been included in interface.py

graph_correction.ipynb is a pipeline including 2 functions (analyze and graph_analyze) from the interface to plot symmetry indices over time during self-correction exercises, in comparison with symmetry indices of manual corrections' positions as targets

recalage_correction.ipynb gathers experiments to register self-correction markers to manual corrections', and inliers RMSE evolution during self-correction exercise, more development of this part to be done in rder to obtain reliable results

icp.ipynb computes .ply pointclouds from xyz data of a sequence, crop the back surface, and calculate the transformation matrix to register two different point clouds of a sequence (tests to compute the distance between segments at different times of a sequence)
Code is gathered in pc_distance.py file

kalman.py contains tests to include kalman filter in markers detection and tracking, has finally not been used