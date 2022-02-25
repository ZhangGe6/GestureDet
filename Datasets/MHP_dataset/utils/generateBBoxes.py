import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random



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

		bounding_boxes_path = '../bounding_boxes'
		if not os.path.exists(bounding_boxes_path):
			os.mkdir(bounding_boxes_path)
		for j in range(len(colorFrames)):
			print (colorFrames[j])
			toks1 = colorFrames[j].split("/")
			toks2 = toks1[3].split("_")
			jointPath = toks1[0]+"/"+toks1[1]+"/"+toks1[2]+"/"+toks2[0]+"_joints.txt"
			print (jointPath)
			points3d = readAnnotation3D(jointPath)[0:21] # the last point is the normal


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


			# compute the minimun Bounding box
			max_x = 0
			min_x = 99999
			max_y = 0
			min_y = 99999
			for k in range(len(points2d)):
				p = points2d[k][0]
				if p[0] > max_x:
					max_x = p[0]
				if p[0] < min_x:
					min_x = p[0]
				if p[1] > max_y:
					max_y = p[1]
				if p[1] < min_y:
					min_y = p[1]
			

			# compute the depth of the centroid of the joints
			p3d_mean = [0,0,0]
			for k in range(len(points3d_cam)):
				p3d_mean[0] += points3d_cam[k][0]
				p3d_mean[1] += points3d_cam[k][1]
				p3d_mean[2] += points3d_cam[k][2]
			p3d_mean[0] /= len(points3d_cam)
			p3d_mean[1] /= len(points3d_cam)
			p3d_mean[2] /= len(points3d_cam)


			# compute the offset considering the depth
			offset = 20 # 20px @ 390mm
			offset = p3d_mean[2]*offset/390
			max_x = int(max_x+offset)
			min_x = int(min_x-offset)
			max_y = int(max_y+offset)
			min_y = int(min_y-offset)
			

			# HERE YOU CAN SAVE points2d, points3d_cam and boundingbox TO A FILE IF YOU WANT
			pathToSaveDir = os.path.join(bounding_boxes_path, "data_"+str(i))
			if not os.path.exists(pathToSaveDir):
				os.mkdir(pathToSaveDir)
			pathToSave = os.path.join(pathToSaveDir, toks2[0]+"_bbox_"+toks2[2].split(".")[0]+".txt")
			print ("Saving bounding box",pathToSave)
			f = open(pathToSave, "w")
			f.write("TOP "+str(int(min_y))+"\n")
			f.write("LEFT "+str(int(min_x))+"\n")
			f.write("BOTTOM "+str(int(max_y))+"\n")
			f.write("RIGHT "+str(int(max_x))+"\n")
			f.close()


			# show a random sample of the sequence
			show = False
			if show and j > rand_idx and j < rand_idx+4:
				img = cv2.imread(colorFrames[j])
				#cv2.circle(img, (int(points2d[k][0][0]), int(points2d[k][0][1])), 5, (255,0,255))
				cv2.rectangle(img, (min_x, min_y), (max_x,max_y), (255,0,0))
				for k in range(points2d.shape[0]):
					cv2.circle(img, (int(points2d[k][0][0]), int(points2d[k][0][1])), 3, (0,0,255))
				cv2.imshow("img", img)
				cv2.waitKey(0)


if __name__ == "__main__":
    main()