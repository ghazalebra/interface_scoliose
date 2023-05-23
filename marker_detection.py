import cv2
import numpy as np
import os
import sys
# import read_single_intensity_raw_file
import csv
from tqdm import tqdm


# the image dimensions
w = 1936
h = 1176
header_size = 512

def read_single_xyz_raw_file(file_path):

    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h, w, 3))

    return data_array

# input is an image (one frame of movement sequence)
def annotate_single_frame(crop, frame, bg=False, frame_xyz=None):

    if bg:
        frame = remove_bg(frame, frame_xyz)

    frame_display, preprocessed_frame = preprocess(frame, crop)

    key_points = detect_markers(preprocessed_frame)
    im_with_key_points = cv2.drawKeypoints(frame_display, key_points, np.array([]), (0, 255, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    key_points = [key_points[j] for j in range(len(key_points))]
    return key_points, im_with_key_points

# path is the path to the folder containing a folder called frames containing the sequence frames
# the function reads the .jpg files inside path/intensity
# reads the xyz .raw files from path/xyz
# The annotated frames will be saved in save_path/annotated_frames
# The landmarks will be saved in save_path/landmarks
def annotate_frames(path, crop):
    i = 0
    intensity_path = path + '/intensity/'
    xyz_path = path + '/xyz/'
    annotated_frame_path = path + '/annotated_frames/'
    os.makedirs(annotated_frame_path, exist_ok=True)
    landmark_path = path + '/landmarks/'
    os.makedirs(landmark_path, exist_ok=True)
    for filename_i, filename_xyz in zip(os.listdir(intensity_path), os.listdir(xyz_path)):
        frame_intensity = cv2.imread(os.path.join(intensity_path, filename_i))
        frame_xyz = read_single_xyz_raw_file(os.path.join(xyz_path, filename_xyz))
        key_points, frame_with_key_points = annotate_single_frame(crop, frame=frame_intensity, bg=True, frame_xyz=frame_xyz)
        cv2.imwrite(annotated_frame_path + 'annotated_frame%d.jpg' % i, frame_with_key_points)
        file = open(landmark_path + 'frame%d_landmarks.txt' % i, 'w+')
        for point in key_points:
            file.write(str(point.pt[0]) + ' ' + str(point.pt[1]) + '\n')
        file.close()
        i += 1


# input is the path to a video
def annotate_video(save_path, video_path):
    cap = cv2.VideoCapture(video_path)

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(frame.shape)
        src_img = frame
        frame = preprocess(frame)
        key_points = detect_markers(frame)
        # print(key_points)
        im_with_key_points = cv2.drawKeypoints(frame, key_points, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_points_x = [key_points[j].pt[0] for j in range(len(key_points))]
        key_points_y = [key_points[j].pt[1] for j in range(len(key_points))]
        key_points_y.sort()
        key_points_x.sort()
        # print(key_points_x)
        # print(key_points_y)
        cv2.imwrite(save_path + 'frame%d.jpg' % i, src_img)
        cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, im_with_key_points)
        break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

def remove_bg_frames(path_intenst, path_xyz):

    i = 1
    frames_path = path + '/intensity/'
    removed_bg_path = path + '/removed_bg_frames/'
    # print(removed_bg_path)
    os.makedirs(removed_bg_path, exist_ok=True)
    for filename in os.listdir(frames_path):
        # finding the index of _I
        index_I = filename.find('_I') + 1
        xyz_file = filename[:index_I] + 'XYZ_' + filename[index_I+2:-4] + '.raw'
        # print(filename)
        # print(xyz_file)
        frame = cv2.imread(os.path.join(frames_path, filename))
        if os.path.exists(os.path.join(path_xyz, xyz_file)):
            frame_xyz = read_single_xyz_raw_file(os.path.join(path_xyz, xyz_file))
        else:
            print(filename, index_I, xyz_file)
            print('filename error!')
            break
        removed_bg_frame = remove_bg(frame, frame_xyz)
        cv2.imwrite(removed_bg_path + 'frame%d.jpg' % i, removed_bg_frame)
        i += 1


def remove_bg(frame, frame_xyz):

    # finding the indices where z > 0 which means the points are not in the background
    keep =  np.where(frame_xyz[:, :, 2]>0)
    processed_frame = np.zeros(frame.shape, dtype=np.float32)
    for i in range(len(keep[0])):
        # the location of the pixel
        # flipping the coordinate vertically to match the frame
        p = (w - keep[1][i] - 1, keep[0][i])
        processed_frame[p] = frame[p]

    return processed_frame

def preprocess(image, crop):
    (w1, w2, h1, h2) = crop
    # removing the background
    # image = remove_bg(image, xyz)

    # cropping the image
    image = image[w1:w2, h1:h2, :]

    

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(3, 3))
    clahe_img = clahe.apply(image_bw)
    # plt.hist(final_img.flat, bins=100, range=(0, 255))
    # plt.show()
    blurred = cv2.medianBlur(clahe_img, 5)
    # ret, threshold = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    circle_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    circles = cv2.erode(255 - blurred, circle_structure, iterations=1)
    circles = cv2.dilate(circles, circle_structure, iterations=2)

    ret, threshold = cv2.threshold(255 - circles, 40, 255, cv2.THRESH_BINARY)

    return clahe_img, 255 - circles


def detect_markers(frame, params=None):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 20
    # params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    # params.maxArea = 40
    #
    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.8
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.8
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.5

    params.minDistBetweenBlobs = 50

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(frame)
    return key_points


def refine_markers(key_points):
    np.sort(key_points)


if __name__ == '__main__':

    path_variants = ['BG/Contraint/Prise01', 'BG/Contraint/Prise02', 'BG/Libre/Prise01', 'BG/Libre/Prise02', 
    'BD/Contraint/Prise01', 'BD/Contraint/Prise02', 'BD/Libre/Prise01', 'BD/Libre/Prise02', 'autocorrection/Prise01', 'autocorrection/Prise02']

    if len(sys.argv) > 1:
        data_path = str(sys.argv[1])
    else:
        data_path = 'Data/Participant1/'
    for path in tqdm(path_variants):
        file_path = data_path + path
        annotate_frames(file_path)
