import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
	matches = []
	for root, dirnames, filenames in os.walk(rootdir):
	  for filename in fnmatch.filter(filenames, pattern):
		  matches.append(os.path.join(root, filename))

	return matches


def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)


def main():

	pathToDataset="../annotated_frames/"


	# iterate sequences
	for i in range(1,22):
		# read the color frames
		path = pathToDataset+"data_"+str(i)+"/"
		colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
		colorFrames = natural_sort(colorFrames)
		print "There are",len(colorFrames),"color frames on the sequence data_"+str(i)
		# read the calibrations for each camera
		print "Loading calibration for ../calibrations/data_"+str(i)
		c_0_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_1/rvec.pkl","r"))
		c_0_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_1/tvec.pkl","r"))
		c_1_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_2/rvec.pkl","r"))
		c_1_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_2/tvec.pkl","r"))
		c_2_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_3/rvec.pkl","r"))
		c_2_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_3/tvec.pkl","r"))
		c_3_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_4/rvec.pkl","r"))
		c_3_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_4/tvec.pkl","r"))

		rand_idx = random.randint(0, len(colorFrames))


		for j in range(len(colorFrames)):
			print colorFrames[j]
			toks1 = colorFrames[j].split("/")
			toks2 = toks1[3].split("_")
			jointPath = toks1[0]+"/"+toks1[1]+"/"+toks1[2]+"/"+toks2[0]+"_joints.txt"
			print jointPath
			points3d = readAnnotation3D(jointPath)[0:21] # the last point is the normal

			# project 3d LM points to the camera coordiante frame
			webcam_id = int(toks2[2].split(".")[0])-1
			print "Calibration for webcam id:",webcam_id
			if webcam_id == 0:
				rvec = c_0_0
				tvec = c_0_1
			elif webcam_id == 1:
				rvec = c_1_0
				tvec = c_1_1
			elif webcam_id == 2:
				rvec = c_2_0
				tvec = c_2_1
			elif webcam_id == 3:
				rvec = c_3_0
				tvec = c_3_1



			R,_ = cv2.Rodrigues(rvec)
			T = np.zeros((4,4))
			for l in range(R.shape[0]):
				for k in range(R.shape[1]):
					T[l][k] = R[l][k]

			for l in range(tvec.shape[0]):
				T[l][3] = tvec[l]
			T[3][3] = 1

			
			points3d_cam = []
			for k in range(len(points3d)):
				p = np.array(points3d[k]).reshape(3,1)
				p = np.append(p, 1).reshape(1,4)
				p_ = np.matmul(T, p.transpose())
				points3d_cam.append(p_)




			# HERE YOU CAN SAVE points3d_cam TO A FILE IS YOU WANT


			# show a random sample of the sequence
			if j > rand_idx and j < rand_idx+4:

				img = cv2.imread(colorFrames[j])
				cv2.imshow("img", img)
				cv2.waitKey(200)

				fig = plt.figure()

				ax1 = fig.add_subplot(211, projection='3d')
				ax1.set_xlim3d(-500,500)
				ax1.set_ylim3d(-500,500)
				ax1.set_zlim3d(0,1000)
				ax1.set_xlabel('X Label')
				ax1.set_ylabel('Y Label')
				ax1.set_zlabel('Z Label')
				ax1.plot([0,0,0],[0,100,0],[0,0,0])
				ax1.plot([100,0,0],[0,0,0],[0,0,0])
				ax1.plot([0,0,0],[0,0,0],[0,0,100])
				ax2 = fig.add_subplot(212, projection='3d')
				ax2.set_xlim3d(-500,500)
				ax2.set_ylim3d(0,1000)
				ax2.set_zlim3d(-500,500)
				ax2.set_xlabel('X Label')
				ax2.set_ylabel('Y Label')
				ax2.set_zlabel('Z Label')
				ax2.plot([0,0,0],[0,100,0],[0,0,0])
				ax2.plot([100,0,0],[0,0,0],[0,0,0])
				ax2.plot([0,0,0],[0,0,0],[0,0,100])		

				for k in range(len(points3d_cam)):
					ax1.scatter(points3d_cam[k][0], points3d_cam[k][1], points3d_cam[k][2], c="red")
					ax2.scatter(points3d[k][0], points3d[k][1], points3d[k][2], c="blue")


				plt.show()


				cv2.waitKey(0)


if __name__ == "__main__":
    main()