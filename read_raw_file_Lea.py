import numpy as np
import cv2
import os
import csv

# the image dimensions
w = 1936
h = 1176
header_size = 512

### FONCTIONS POUR LE TRAITEMENT DES IMAGES ###
def equalize_histogram(img):
    [M, N]=img.shape
    # Calcul de la transformation
    T=1/(M*N)*np.cumsum(np.histogram(img, bins=256, range=(0,255))[0])
    img_eq=255*T[img]
    return img_eq

# reads the intensity raw file and converts it to an image_like array
def read_single_intensity_raw_file(file_path):
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h, w)).T
    # reversing the image vertically
    data_array = data_array[-1:0:-1, :]
    # normalizing the values to be in the range (0, 255)
    data_array = 255 * (data_array - np.min(data_array)) / np.max(data_array)
    return data_array


# reads all the intensity raw files in a folder and save them as images in save_path folder (all the frames of an acquisition)
def read_raw_intensity_frames(folder_path):
    for filename in os.listdir(folder_path):
        # check if it's an intensity file
        if '_I_' in filename:
            frame = read_single_intensity_raw_file(os.path.join(folder_path, filename))
            # removes the '.raw' extension from the end of the filename and replaces it with '.jpg'
            cv2.imwrite(folder_path + '/intensity/' + filename[:-4] + '.jpg', frame)

# reads the xyz raw file and returns coordinates for detected markers on the corresponding image
def read_single_xyz_raw_file(file_path, marqueurs):
    # the image dimensions
    w = 1936
    h = 1176
    header_size = 512
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h*w,3))
    #retrieve the depth as an image (and flip upside down)
    x_array = data_array[:,0].reshape((1176,1936)).T
    x_array = x_array[-1:0:-1, :][800:1400, 350:850]
    y_array = data_array[:,1].reshape((1176,1936)).T
    y_array = y_array[-1:0:-1, :][800:1400, 350:850]
    z_array = data_array[:,2].reshape((1176,1936)).T
    z_array = z_array[-1:0:-1, :][800:1400, 350:850]

    coordos = []
    for el in marqueurs:
        x = (x_array[int(el[1]),int(el[0])])
        y = (y_array[int(el[1]),int(el[0])])
        z = (z_array[int(el[1]),int(el[0])])
        coordos.append([x, y, z]) #liste de 5 listes contenant les coordos (x,y,z) pour chaque marqueur
    
    return coordos

#print(read_single_xyz_raw_file('C:/Users/LEA/Desktop/Poly/H2023/Projet 3/Data/Participant01/autocorrection/Prise02/auto_02_015454_XYZ_0.raw'))

# reads all the xyz raw files in a folder and save the z images in save_path folder (all the frames of an acquisition)
def read_raw_xyz_frames(folder_path, dict_coordo):
    xyz_path = folder_path + '/xyz'
    for filename in os.listdir(xyz_path):
        marqueurs = dict_coordo[f'image{int(filename[19:-4])+1}']
        # trouve les coordonnées associées aux marqueurs détectés
        coordos = read_single_xyz_raw_file(os.path.join(xyz_path, filename), marqueurs)
        # removes the '.raw' extension from the end of the filename and replaces it with '.png'
        csv_filename = folder_path + '/XYZ_converted/' + filename[:-4] + '.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for c in coordos:
                writer.writerow(c)

'''
folder_path = 'C:/Users/LEA/Desktop/Poly/H2023/Projet 3/Data/Participant01/autocorrection/Prise02/'
save_path = 'C:/Users/LEA/Desktop/Poly/H2023/Projet 3/Data/Participant01/autocorrection/Prise02/Converted/'
save_xyz = 'C:/Users/LEA/Desktop/Poly/H2023/Projet 3/Data/Participant01/autocorrection/Prise02/XYZ_converted/'

marqueurs = [[233.58773803710938, 413.35284423828125], [223.70643615722656, 312.5287170410156], [281.0943603515625, 235.71827697753906], [226.80198669433594, 60.87376022338867], [163.41574096679688, 247.35049438476562]]
read_raw_xyz_frames(folder_path, marqueurs)
'''
