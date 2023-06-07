import cv2
import numpy as np
import json
import pylab as pl


jsonfile = open(r'D:\StageE23\Data\Ete_2022\Participant06\autocorrection\Prise01\dict\positions.json', 'r')
dict_coordo = json.load(jsonfile)

x = [int(key) for key in dict_coordo.keys()]
observations = np.array(list(dict_coordo.values()), dtype=np.float32)
print('obs:', observations)
n_im = len(x)

kalman = cv2.KalmanFilter(4, 2, 0)

kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#kalman.processNoiseCov = 1e-5 * np.eye(4)
kalman.measurementNoiseCov = 1e-5 * np.ones((2, 2), np.float32)
kalman.errorCovPost = 1. * np.ones((4, 4), np.float32)
kalman.measurementMatrix = np.ones((1, 4), np.float32)

filtered_state_means = np.zeros((n_im, 2), np.float32)

for im, o in enumerate(observations):
    prediction = kalman.predict()
    filtered_state_means[im] = kalman.correct(o)

pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, filtered_state_means[:, 0],
                        linestyle='-', marker='o', color='r',
                        label='position est.')
velocity_line = pl.plot(x, filtered_state_means[:, 1],
                        linestyle='-', marker='o', color='g',
                        label='velocity est.')
pl.legend(loc='lower right')
pl.legend(loc='lower right')
pl.xlabel('time')
pl.show()