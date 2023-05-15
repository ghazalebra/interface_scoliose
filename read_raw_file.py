import numpy as np

import cv2
import os
import csv
from tqdm import tqdm
import shutil

# the image dimensions
w = 1936
h = 1176
header_size = 512

# folder_path = '/home/travail/ghebr/Data/Participant01/autocorrection/Prise01'
# path_variants = ['']
# save_path = '/home/travail/ghebr/project/Data/Participant1/autocorrection/take1/'
# os.makedirs(save_path, exist_ok=True)


# reads the intensity raw file and converts it to an image_like array
def read_single_intensity_raw_file(file_path):
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h, w)).T
    
    # reversing the image vertically
    data_array = np.flip(data_array, 0)
    # normalizing the values to be in the range (0, 255)
    data_array = 255 * (data_array - np.min(data_array)) / np.max(data_array)
    return data_array


def read_single_xyz_raw_file(file_path):
    with open(file_path, 'r') as f:
        f.seek(header_size)
        # data = f.read()
        data_array = np.fromfile(f, np.float32).reshape((h, w, 3))
    
    # reversing the image vertically
    # data_array = data_array[0:1:1, :]
    # normalizing the values to be in the range (0, 255)
    # data_array = 255 * (data_array - np.min(data_array)) / np.max(data_array)
    return data_array


# reads all the intensity raw files in a folder and save them as images in save_path folder (all the frames of an acquisition)
def read_raw_intensity_frames(folder_path, save_path):
    for filename in os.listdir(folder_path):
        # print('he')
        # check if it's an intensity file
        if '_I_' in filename:
            # print('ha')
            frame = read_single_intensity_raw_file(os.path.join(folder_path, filename))
            # removes the '.raw' extension from the end of the filename and replaces it with '.jpg'
            cv2.imwrite(save_path + filename[:-4] + '.jpg', frame)

def read_raw_xyz_frames(folder_path, save_path):
    for filename in os.listdir(folder_path):
        # print('found xyz file!')
        # check if it's an xyz file
        if '_XYZ_' in filename:
            coordinates = read_single_xyz_raw_file(os.path.join(folder_path, filename))
            # removes the '.raw' extension from the end of the filename and replaces it with '.txt'
            save_file_path = save_path + filename[:-4] + '.csv'
            with open(save_file_path, 'w+') as xyz_file:
                xyz_file.write(str(coordinates))

def copy_xyz_frames(src_path, des_path):
    for filename in os.listdir(src_path):
        # print('found xyz file!')
        # check if it's an xyz file
        if '_XYZ_' in filename:
            file_path = os.path.join(src_path, filename)
            save_file_path = os.path.join(des_path, filename)
            # print(file_path, save_file_path)
            shutil.copy(file_path, save_file_path)

def write_xyz_coordinates(folder_path, dict_coordo):
    xyz_path = folder_path + '/xyz/'
    for filename in os.listdir(xyz_path):
        marqueurs = dict_coordo[f'image{int(filename[19:-4])+1}']
        # trouve les coordonnées associées aux marqueurs détectés
        coordos = find_xyz_coordinates(os.path.join(xyz_path, filename), marqueurs)
        # removes the '.raw' extension from the end of the filename and replaces it with '.png'
        csv_filename = folder_path + '/XYZ_converted/' + filename[:-4] + '.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for c in coordos:
                writer.writerow(c)

def find_xyz_coordinates(file_path, marqueurs):
    # the image dimensions
    w = 1936
    h = 1176
    header_size = 512
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h,w,3))
    #retrieve the depth as an image (and flip upside down)
    x_array = data_array[:, :,0].T
    x_array = x_array[-1:0:-1, :][800:1400, 350:850]
    y_array = data_array[:, :,1].T
    y_array = y_array[-1:0:-1, :][800:1400, 350:850]
    z_array = data_array[:, :,2].T
    z_array = z_array[-1:0:-1, :][800:1400, 350:850]

    coordos = []
    for el in marqueurs:
        x = (x_array[round(el[1]),round(el[0])])
        y = (y_array[round(el[1]),round(el[0])])
        z = (z_array[round(el[1]),round(el[0])])
        coordos.append([x, y, z]) #liste de 5 listes contenant les coordos (x,y,z) pour chaque marqueur
    return coordos
'''
path_variants = ['autocorrection/Prise01', 'autocorrection/Prise02']
# path_variants = ['BG/Libre/Prise02']
for i in tqdm(range(1, 15)):
    save_path_root = 'C:/Users/LEA/Desktop/Poly/H2023/Projet3/Data2/Ete_2022/Participant0' + str(i) + '/'
    if i < 10:
        for path in path_variants:
            folder_path = 'C:/Users/LEA/Desktop/Poly/H2023/Projet3/Data/Ete_2022/Participant0' + str(i) + '/'
            save_path = save_path_root
            folder_path += path + '/'
            save_path1 = save_path + path + '/intensity/'
            save_path2 = save_path + path + '/xyz/'
            
            #shutil.rmtree(save_path2)
            os.makedirs(save_path1, exist_ok=True)
            os.makedirs(save_path2, exist_ok=True)
            read_raw_intensity_frames(folder_path, save_path1)

            # read_raw_xyz_frames(folder_path, save_path2)
            copy_xyz_frames(folder_path, save_path2)
            # print(folder_path)
            # print(save_path)
        
    else:
        for path in path_variants:
            folder_path = 'C:/Users/LEA/Desktop/Poly/H2023/Projet3/Data/Ete_2022/Participant' + str(i) + '/'
            save_path = save_path_root
            folder_path += path + '/'
            save_path1 = save_path + path + '/intensity/'
            save_path2 = save_path + path + '/xyz/'
            #shutil.rmtree(save_path2)
            os.makedirs(save_path1, exist_ok=True)
            os.makedirs(save_path2, exist_ok=True)
            
            copy_xyz_frames(folder_path, save_path2)
            read_raw_intensity_frames(folder_path, save_path1)
            # read_raw_xyz_frames(folder_path, save_path2)
            # print(folder_path)
            # print(save_path)
    
'''



