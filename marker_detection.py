import cv2
import numpy as np
import os
import sys
import tqdm
import matplotlib.pyplot as plt

# input is an image (one frame of movement sequence)
def annotate_single_frame(frame, w1=300, w2=1500, h1=350, h2=850):

    frame_display, preprocessed_frame = preprocess(frame, w1, w2, h1, h2)

    key_points = detect_markers(preprocessed_frame)
    im_with_key_points = cv2.drawKeypoints(frame_display, key_points, np.array([]), (0, 255, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    key_points = [key_points[j] for j in range(len(key_points))]
    return key_points, im_with_key_points


# path is the path to the folder containing a folder called frames containing the sequence frames
# the function reads the files inside path/frames
# The annotated frames will be saved in save_path/annotated_frames
# The landmarks will be saved in save_path/landmarks
def annotate_frames(path, w1=300, w2=1500, h1=350, h2=850):
    i = 0
    intensity_path = path + '/intensity_removed_bg/'
    annotated_frame_path = path + '/annotated_frames/'
    landmark_path = path + '/landmarks/'
    os.makedirs(landmark_path, exist_ok=True)
    for filename_i in os.listdir(intensity_path):
        frame_intensity = cv2.imread(os.path.join(intensity_path, filename_i))
        key_points, frame_with_key_points = annotate_single_frame(frame=frame_intensity, w1=w1, w2=w2, h1=h1, h2=h2)
        index_I = filename_i.find('_I') + 1
        annotated_file = filename_i[:index_I] + 'annotated_' + filename_i[index_I+2:]
        landmarks_file = filename_i[:index_I] + 'landmarks_' + filename_i[index_I+2:]
        cv2.imwrite(annotated_frame_path + annotated_file, frame_with_key_points)
        file = open(landmark_path + landmarks_file, 'w+')
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
        print(frame.shape)
        print(frame.shape)
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
        print(key_points_x)
        print(key_points_y)
        print(key_points_x)
        print(key_points_y)
        cv2.imwrite(save_path + 'frame%d.jpg' % i, src_img)
        cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, im_with_key_points)
        break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

def preprocess(image, w1=300, w2=1500, h1=350, h2=850):

    # removing the background
    # image = remove_bg(image, xyz)

    # cropping the image
    image = image[w1:w2, h1:h2, :]

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(3, 3))
    clahe_img = clahe.apply(image_bw)
    # plt.hist(final_img.flat, bins=100, range=(0, 255))
    # plt.show()
    blurred = cv2.medianBlur(clahe_img, 5)
    # ret, threshold = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    circle_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    circles = cv2.erode(255 - clahe_img, circle_structure, iterations=1)
    circles = cv2.dilate(circles, circle_structure, iterations=2)

    ret, threshold = cv2.threshold(255 - circles, 40, 255, cv2.THRESH_BINARY)

    return clahe_img, 255 - circles


def detect_markers(frame, params=None):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 20
    #params.maxThreshold = 60
    #params.thresholdStep = 20

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 78
    params.maxArea = 310
    #
    # Filter by Circularity
    #params.filterByCircularity = True
    #params.minCircularity = 0.8
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.8
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.5

    params.minDistBetweenBlobs = 70

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(frame)
    return key_points

if __name__ == '__main__':

    path_variants = ['autocorrection/Prise01', 'autocorrection/Prise02',
    # 'BG/Contraint/Prise01', 'BG/Contraint/Prise02', 'BG/Libre/Prise01', 'BG/Libre/Prise02', 
    # 'BD/Contraint/Prise01', 'BD/Contraint/Prise02', 'BD/Libre/Prise01', 'BD/Libre/Prise02'
    ]

    if len(sys.argv) > 1:
        number = str(sys.argv[1])
    else:
        number = '1'
    data_path = '/home/travail/ghebr/project/Data/Participant' + number + '/'

    for path in tqdm(path_variants):
        file_path = data_path + path
        annotate_frames(file_path)
    # frame_intensity = cv2.imread(file_path)
    # frame_xyz = read_single_xyz_raw_file(xyz_path)
    # markers, annotated_frame = annotate_single_frame(frame=frame_intensity, frame_xyz=frame_xyz, bg=True, w1=700, w2=1220)
    # cv2.imshow('image', annotated_frame)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() 