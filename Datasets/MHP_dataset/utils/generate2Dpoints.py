import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random

def saveAnnotation(jointCamPath, positions):
	fOut = open(jointCamPath, 'w+')
	fOut.write("F4_KNU1_A " + str(positions[0][0]) + " " + str(positions[0][1]) + "\n")
	fOut.write("F4_KNU1_B " + str(positions[1][0]) + " " + str(positions[1][1]) + "\n")
	fOut.write("F4_KNU2_A " + str(positions[2][0]) + " " + str(positions[2][1]) + "\n")
	fOut.write("F4_KNU3_A " + str(positions[3][0]) + " " + str(positions[3][1]) + "\n")

	fOut.write("F3_KNU1_A " + str(positions[4][0]) + " " + str(positions[4][1]) + "\n")
	fOut.write("F3_KNU1_B " + str(positions[5][0]) + " " + str(positions[5][1]) + "\n")
	fOut.write("F3_KNU2_A " + str(positions[6][0]) + " " + str(positions[6][1]) + "\n")
	fOut.write("F3_KNU3_A " + str(positions[7][0]) + " " + str(positions[7][1]) + "\n")

	fOut.write("F1_KNU1_A " + str(positions[8][0]) + " " + str(positions[8][1]) + "\n")
	fOut.write("F1_KNU1_B " + str(positions[9][0]) + " " + str(positions[9][1]) + "\n")
	fOut.write("F1_KNU2_A " + str(positions[10][0]) + " " + str(positions[10][1]) + "\n")
	fOut.write("F1_KNU3_A " + str(positions[11][0]) + " " + str(positions[11][1]) + "\n")

	fOut.write("F2_KNU1_A " + str(positions[12][0]) + " " + str(positions[12][1]) + "\n")
	fOut.write("F2_KNU1_B " + str(positions[13][0]) + " " + str(positions[13][1]) + "\n")
	fOut.write("F2_KNU2_A " + str(positions[14][0]) + " " + str(positions[14][1]) + "\n")
	fOut.write("F2_KNU3_A " + str(positions[15][0]) + " " + str(positions[15][1]) + "\n")

	fOut.write("TH_KNU1_A " + str(positions[16][0]) + " " + str(positions[16][1]) + "\n")
	fOut.write("TH_KNU1_B " + str(positions[17][0]) + " " + str(positions[17][1]) + "\n")
	fOut.write("TH_KNU2_A " + str(positions[18][0]) + " " + str(positions[18][1]) + "\n")
	fOut.write("TH_KNU3_A " + str(positions[19][0]) + " " + str(positions[19][1]) + "\n")
	fOut.write("PALM_POSITION " + str(positions[20][0]) + " " + str(positions[20][1]) + "\n")
	fOut.close()

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
	f = open(file, "rb")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)

def getCameraMatrix():
	Fx = 614.878
	Fy = 615.479
	Cx = 313.219
	Cy = 231.288
	cameraMatrix = np.array([[Fx, 0, Cx],
					[0, Fy, Cy],
					[0, 0, 1]])
	return cameraMatrix

def getDistCoeffs():
	return np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])



def main():

	pathToDataset="../annotated_frames/"

	cameraMatrix = getCameraMatrix()
	distCoeffs = getDistCoeffs()

	# iterate sequences
	for i in range(1,22):
		# read the color frames
		path = pathToDataset+"data_"+str(i)+"/"
		colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
		colorFrames = natural_sort(colorFrames)
		print ("There are",len(colorFrames),"color frames on the sequence data_"+str(i))
		# read the calibrations for each camera
		print ("Loading calibration for ../calibrations/data_"+str(i))
		c_0_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_1/rvec.pkl","rb"), encoding='latin1')
		c_0_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_1/tvec.pkl","rb"), encoding='latin1')
		c_1_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_2/rvec.pkl","rb"), encoding='latin1')
		c_1_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_2/tvec.pkl","rb"), encoding='latin1')
		c_2_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_3/rvec.pkl","rb"), encoding='latin1')
		c_2_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_3/tvec.pkl","rb"), encoding='latin1')
		c_3_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_4/rvec.pkl","rb"), encoding='latin1')
		c_3_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_4/tvec.pkl","rb"), encoding='latin1')

		rand_idx = random.randint(0, len(colorFrames))

		projections_2d_path = '../projections_2d'
		if not os.path.exists(projections_2d_path):
			os.mkdir(projections_2d_path)
		for j in range(len(colorFrames)):
			print (colorFrames[j])
			toks1 = colorFrames[j].split("/")
			toks2 = toks1[3].split("_")
			jointPath = toks1[0]+"/"+toks1[1]+"/"+toks1[2]+"/"+toks2[0]+"_joints.txt"
			print (jointPath)
			points3d = readAnnotation3D(jointPath)[0:21] # the last point is the normal

			# project 3d LM points to the image plane
			webcam_id = int(toks2[2].split(".")[0])-1
			print ("Calibration for webcam id:",webcam_id)
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

			points2d, _ = cv2.projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs)


			# HERE YOU CAN SAVE points2d TO A FILE IF YOU WANT
			pathToSaveDir = os.path.join(projections_2d_path, "data_"+str(i))
			if not os.path.exists(pathToSaveDir):
				os.mkdir(pathToSaveDir)
			pathToSave = os.path.join(pathToSaveDir, toks2[0]+"_jointsCam_"+toks2[2].split(".")[0]+".txt")
			print ("Saving 2d projections",pathToSave)
			saveAnnotation(pathToSave, np.array(points2d).reshape(21,2))



			# show a random sample of the sequence
			show = False
			if show and j > rand_idx and j < rand_idx+4:
				img = cv2.imread(colorFrames[j])
				for k in range(points2d.shape[0]):
					cv2.circle(img, (int(points2d[k][0][0]), int(points2d[k][0][1])), 3, (0,0,255))
				cv2.imshow("img", img)
				cv2.waitKey(0)


if __name__ == "__main__":
    main()