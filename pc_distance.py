import os
import cv2
from tqdm import tqdm
import numpy as np
import math
import open3d as o3d
import preprocess as PRE
import copy
import csv

# Crée les point clouds pour une séquence et les enregistre dans path + '/xyz_removed_bg/'
def create_pc(path):
    save_path2 = path + '/xyz_removed_bg/'
    intensity_path = path + '/intensity/'
    xyz_path = path + '/xyz/'

    os.makedirs(save_path2, exist_ok=True)

    for filename in tqdm(os.listdir(intensity_path)):
        index_I = filename.find('_I') + 1
        xyz_file = filename[:index_I] + 'XYZ_' + filename[index_I+2:-4] + '.raw'
        image_path = os.path.join(intensity_path, filename)
        pc_path = os.path.join(xyz_path, xyz_file)
        xyz = PRE.read_single_xyz_raw_file(pc_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        removed_bg_points = PRE.remove_bg(xyz, image)[0]

        pc_removed_bg = o3d.geometry.PointCloud()
        pc_removed_bg.points = o3d.utility.Vector3dVector(removed_bg_points)
        cl, ind = pc_removed_bg.remove_statistical_outlier(nb_neighbors=40,
                                                std_ratio=1.5)
        inlier_removed_bg = pc_removed_bg.select_by_index(ind)

        save_xyz_path = os.path.join(save_path2, xyz_file[:-4] + '.ply')
        o3d.io.write_point_cloud(save_xyz_path, inlier_removed_bg)


# Rogne le point cloud pour ne garder que le dos

# !!! AJUSTEMENTS À FAIRE SELON PARTICIPANTS, TRAVAILLER SUR GÉNÉRALISATION !!!
def crop_pc(pc):
    pc_array = np.asarray(pc.points)

    legs = np.where(pc_array[:,0] < -300)
    head = np.where(pc_array[:,0] > np.max(pc_array[:,0])-300)
    armR = np.where(pc_array[:,1] > np.max(pc_array[:,1])-90)
    armL = np.where(pc_array[:,1] < np.min(pc_array[:,1])+90)

    mask = np.ones((pc_array.shape[0],3), dtype=bool)
    mask[legs] = False
    mask[head] = False
    mask[armR] = False
    mask[armL] = False
    #mask.reshape((500123, 3))

    back = np.array(pc_array[mask], dtype=np.float64)

    pc = np.array(back.reshape((back.shape[0]//3, 3)))

    pc_cropped = o3d.geometry.PointCloud()
    pc_cropped.points = o3d.utility.Vector3dVector(pc)
    
    return pc_cropped


def IsRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def RotationMatrixToEulerAngles(R):
    assert(IsRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        y = math.atan2(R[2,1], R[2,2])
        x = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        y = math.atan2(-R[1,2], R[1,1])
        x = math.atan2(-R[2,0], R[1,1])
        z = 0

    return np.array([np.degrees(x),np.degrees(y),np.degrees(z)])

def icp_values(path_pc, target_nb):
    trans_init = np.identity(4)
    body_target = crop_pc(o3d.io.read_point_cloud(os.path.join(path_pc, os.listdir(path_pc)[target_nb-1])))

    ICP = {}
    for i, filename in enumerate(os.listdir(path_pc)):
        pc = o3d.io.read_point_cloud(os.path.join(path_pc, filename))
        body = crop_pc(pc)
        transfo = o3d.pipelines.registration.registration_icp(body, body_target, 10, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        ICP.update({i+1:{'transfo_matrix':transfo.transformation}}) 
    
    for im, icp in ICP.items():
        RST = icp['transfo_matrix']
        T = RST[0:3, 3]
        len_T = np.linalg.norm(T)
        ICP[im].update({'translation_distance':len_T})

        axis_T = T/len_T
        ICP[im].update({'translation_axis':axis_T})

        RS = RST[0:3, 0:3]
        S = np.zeros((3,3))
        S[0,0] = np.linalg.norm(RS[:,0])
        S[1,1] = np.linalg.norm(RS[:,1])
        S[2,2] = np.linalg.norm(RS[:,2])
        R = np.matmul(RS, np.linalg.inv(S)) #si S est la matrice identité, R=RS et aucun scaling
        
        [y,x,z] = RotationMatrixToEulerAngles(R)
        ICP[im].update({'rotation_x' : x})
        ICP[im].update({'rotation_y' : y})
        ICP[im].update({'rotation_z' : z})

        cos_rot = (np.trace(R)-1)/2
        angle_rot = np.degrees(np.arccos(cos_rot))
        ICP[im].update({'rotation_angle': angle_rot})

        axis_rot = np.multiply(1/np.sqrt((3-np.trace(R))*(1+np.trace(R))), np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]))
        ICP[im].update({'rotation_axis': axis_rot})

    return ICP

def saveICP(path, target_nb):
    if '/xyz_removed_bg/' not in os.listdir(path):
        create_pc(path)
    elif len(os.listdir(path+'/xyz_removed_bg/')) == 0:
        create_pc(path)

    ICP = icp_values(path + r'\xyz_removed_bg', target_nb)

    os.makedirs(path + '/ICP/', exist_ok=True)
    with open(path + r'\ICP\icp.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        entete = ['image no']
        for el in ICP[1].keys():
            entete += [el]
        writer.writerow(entete)
        for im, vals in ICP.items():
            row = [im]
            for val in vals.values():
                row += [val]
            writer.writerow(row)


saveICP(r'D:\StageE23\Data\Ete_2022\Participant08\autocorrection\Prise02', 40)


# Fonction de http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html pour visualiser le résultat de l'ICP algorithm
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])