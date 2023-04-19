import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# input is an image (one frame of movement sequence)
def annotate_single_frame(frame):
    frame_display, preprocessed_frame = preprocess(frame)
    key_points = detect_markers(preprocessed_frame)
    im_with_key_points = cv2.drawKeypoints(frame_display, key_points, np.array([]), (0, 255, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    key_points = [key_points[j] for j in range(len(key_points))]
    return key_points, im_with_key_points


# path is the path to the folder containing a folder called frames containing the sequence frames
# the function reads the files inside path/frames
# The annotated frames will be saved in save_path/annotated_frames
# The landmarks will be saved in save_path/landmarks
def annotate_frames(path):
    i = 0
    frames_path = path + '/frames/'
    annotated_frame_path = path + '/annotated_frames/'
    landmark_path = path + '/landmarks/'
    for filename in os.listdir(frames_path):
        frame = cv2.imread(os.path.join(frames_path, filename))
        key_points, frame_with_key_points = annotate_single_frame(frame)
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
        cv2.imwrite(save_path + 'frame%d.jpg' % i, src_img)
        cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, im_with_key_points)
        break

        i += 1

    cap.release()
    cv2.destroyAllWindows()


def preprocess(image):
    image = image[800:1400, 350:850]

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

    if len(sys.argv) > 1:
        data_path = str(sys.argv[1])
    else:
        data_path = r'C:\Users\LEA\Desktop\Poly\H2023\Projet 3\Data\Participant01\autocorrection\Prise01\Converted'
    #annotate_frames(data_path)
    key_points, im_with_key_points = annotate_single_frame(cv2.imread(data_path+r'\auto_01_014763_I_1.jpg'))
    print(key_points)
    cv2.imwrite(data_path+r'\annot.jpg', im_with_key_points)
