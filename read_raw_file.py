import numpy as np
from scipy.ndimage import median_filter
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

def read_single_xyz_raw_file(file_path):
    with open(file_path, 'r') as f:
        f.seek(header_size)
        # data = f.read()
        frame = np.fromfile(f, np.float32).reshape((h, w, 3))
        #retrieve the depth as an image (and flip upside down)
        z = frame[:,:,2].T
        z = z[-1:0:-1, :]
        zz = z[np.where(z>0)]
        body_z = np.median(zz)
        z_nobg = z
        z_nobg[np.where(z > body_z + 600)] = 0
        z_nobg = median_filter(z_nobg, 5) #correction pour trous dans l'image
    # reversing the image vertically
    # data_array = data_array[0:1:1, :]
    # normalizing the values to be in the range (0, 255)
    # data_array = 255 * (data_array - np.min(data_array)) / np.max(data_array)
    return frame, z_nobg

def read_raw_xyz_frames(folder_path, save_path):
    for filename in os.listdir(folder_path):
        # print('found xyz file!')
        # check if it's an xyz file
        if '_XYZ_' in filename:
            frame, z_nobg = read_single_xyz_raw_file(os.path.join(folder_path, filename))
            
            # removes the '.raw' extension from the end of the filename and replaces it with '.txt'
            cv2.imwrite(save_path + filename[:-4] + '.png', z_nobg)

#path = r'D:\StageE23\Data\Ete_2022\Participant07\autocorrection\Prise02'
#read_raw_xyz_frames(path, path+'\\xyz_nobg\\')


def copy_xyz_frames(src_path, des_path):
    for filename in os.listdir(src_path):
        # print('found xyz file!')
        # check if it's an xyz file
        if '_XYZ_' in filename:
            file_path = os.path.join(src_path, filename)
            save_file_path = os.path.join(des_path, filename)
            # print(file_path, save_file_path)
            shutil.copy(file_path, save_file_path)

def write_xyz_coordinates(folder_path, dict_coordo, crop):
    xyz_path = folder_path + '/xyz/'
    for filename in os.listdir(xyz_path):
        i = filename.index('XYZ')+4
        marqueurs = dict_coordo[f'image{int(filename[i:-4])+1}']
        # trouve les coordonnées associées aux marqueurs détectés
        coordos = find_xyz_coordinates(os.path.join(xyz_path, filename), marqueurs, crop)
        # removes the '.raw' extension from the end of the filename and replaces it with '.png'
        csv_filename = folder_path + '/XYZ_converted/' + filename[:-4] + '.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for c in coordos:
                writer.writerow(c)

def find_xyz_coordinates(file_path, marqueurs, crop):
    # the image dimensions
    w = 1936
    h = 1176
    header_size = 512
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h,w,3))
    #retrieve the depth as an image (and flip upside down)
    (w1, w2, h1, h2) = crop
    x_array = data_array[:, :,0].T
    x_array = x_array[-1:0:-1, :][h1:h2, w1:w2]
    y_array = data_array[:, :,1].T
    y_array = y_array[-1:0:-1, :][h1:h2, w1:w2]
    z_array = data_array[:, :,2].T
    z_array = z_array[-1:0:-1, :][h1:h2, w1:w2]

    coordos = []
    for el in marqueurs:
        x = (x_array[round(el[1]),round(el[0])])
        y = (y_array[round(el[1]),round(el[0])])
        z = (z_array[round(el[1]),round(el[0])])

        i = -2
        while [x,y,z] == [0.0, 0.0, 0.0]:
            print(i)
            x = (x_array[round(el[1]+i),round(el[0]+i)])
            print(x)
            y = (y_array[round(el[1]+i),round(el[0]+i)])
            print(y)
            z = (z_array[round(el[1]+i),round(el[0]+i)])
            print(z)
            i += 1

        coordos.append([x, y, z]) #liste de 5 listes contenant les coordos (x,y,z) pour chaque marqueur

    return coordos