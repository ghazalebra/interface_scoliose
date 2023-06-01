import open3d as o3d
import numpy as np
import json
import cv2
from tqdm import tqdm
import os
import sys



# the image dimensions
w = 1936
h = 1176
header_size = 512

def read_single_xyz_raw_file(file_path):

    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h, w, 3))
    data_array = np.flip(data_array, 1)
    # first index is horizental, the seocnd index in inverted
    data_array_modified = data_array.reshape((-1, 3))

    return data_array_modified



# modified it to work with point clouds
def remove_bg(pc, image):
    keep =  np.where(pc[:, 2]>0)
    # h
    keep1 = keep[0] // w
    # w
    keep2 = keep[0] % w
    pc_removed_0 = pc[keep[0]]
    image_no_bg = np.zeros(image.shape, dtype=np.float32)
    for i in tqdm(range(len(keep[0]))):
        p = (keep2[i], keep1[i])
        image_no_bg[p] = image[p]
    
    z_body = np.median(pc_removed_0[:, 2])
    z_bg = np.max(pc_removed_0[:, 2])
    z_front = np.min(pc_removed_0[:, 2])
    non_bg = np.where(pc_removed_0[:, 2] < (z_body + z_bg) / 2.)
    pc_no_bg = pc_removed_0[non_bg[0]]
    return pc_no_bg, image_no_bg


if __name__ == '__main__':

    path_variants = ['BG/Contraint/Prise01', 'BG/Contraint/Prise02', 'BG/Libre/Prise01', 'BG/Libre/Prise02', 
    'BD/Contraint/Prise01', 'BD/Contraint/Prise02', 'BD/Libre/Prise01', 'BD/Libre/Prise02', 'autocorrection/Prise01', 'autocorrection/Prise02']


    if len(sys.argv) > 1:
        number = str(sys.argv[1])
    else:
        number = '1'
    data_path = '/home/travail/ghebr/project/Data/Participant' + number + '/'
    
    for path in tqdm(path_variants):
        save_path = data_path
        intensity_path = data_path + path + '/intensity/'
        xyz_path = data_path + path + '/xyz/'
        save_path1 = save_path + path + '/intensity_removed_bg/'
        save_path2 = save_path + path + '/xyz_removed_bg/'
        # shutil.rmtree(save_path2)
        os.makedirs(save_path1, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
        for filename in tqdm(os.listdir(intensity_path)):
            index_I = filename.find('_I') + 1
            xyz_file = filename[:index_I] + 'XYZ_' + filename[index_I+2:-4] + '.raw'
            image_path = os.path.join(intensity_path, filename)
            pc_path = os.path.join(xyz_path, xyz_file)
            xyz = read_single_xyz_raw_file(pc_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            removed_bg_points, image_no_bg = remove_bg(xyz, image)
            
            save_image_path = os.path.join(save_path1, filename)
            cv2.imwrite(save_image_path, image_no_bg)

            pc_removed_bg = o3d.geometry.PointCloud()
            pc_removed_bg.points = o3d.utility.Vector3dVector(removed_bg_points)
            cl, ind = pc_removed_bg.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
            inlier_removed_bg = pc_removed_bg.select_by_index(ind)

            save_xyz_path = os.path.join(save_path2, xyz_file[:-4] + '.ply')
            o3d.io.write_point_cloud(save_xyz_path, inlier_removed_bg)

    