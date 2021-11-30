from mylib import config, thread
from mylib.yolo import YOLO
from mylib.mailer import Mailer
from mylib.detection import detect_people
from mylib.birdview import compute_perspective_transform,compute_point_perspective_transformation
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, cv2, os, time , imutils ,schedule #,yaml
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import requests


COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (255, 165, 0)
COLOR_GREY = (200, 200, 200)
BIG_CIRCLE = 30
SMALL_CIRCLE = 3


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")

ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
	
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([config.PEOPLE_MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.PEOPLE_MODEL_PATH, "yolo-fastest.weights"])
#weightsPath = os.path.sep.join([config.PEOPLE_MODEL_PATH, "yolov4-tiny.weights"])
configPath = os.path.sep.join([config.PEOPLE_MODEL_PATH, "yolo-fastest.cfg"])
#configPath = os.path.sep.join([config.PEOPLE_MODEL_PATH, "yolov4-tiny.cfg"])

classes = ["good", "bad", "none"]
#yolo = YOLO("models/yolov4-tiny.cfg", "models/yolov4-tiny.weights", classes)
yolo = YOLO(os.path.sep.join([config.FACE_MODEL_PATH,"yolo-fastest.cfg"]), os.path.sep.join([config.FACE_MODEL_PATH,"yolo-fastest.weights"]), classes)
#yolo.size = int(args.size)
yolo.size = 416
#yolo.confidence = float(args.confidence)
yolo.confidence = 0.5
colors = [COLOR_GREEN, COLOR_YELLOW, COLOR_RED]
count = 0

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("")
	print("[提示] Looking for GPU")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
#lna = list()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
"""
for i in net.getUnconnectedOutLayers():
	lna.append(ln[[i][0]-1])
ln = lna
del lna
"""
# if a video path was not supplied, grab a reference to the camera
if not args.get("input", False):
	print("[提示] Starting the live stream..")
	vs = cv2.VideoCapture(config.url)
	if config.Thread:
			cap = thread.ThreadingClass(config.url)
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[提示] Starting the video..")
	vs = cv2.VideoCapture(args["input"])
	if config.Thread:
			cap = thread.ThreadingClass(args["input"])

# start the FPS counter
fps = FPS().start()
#matplotlib figure size
fig, ax1 = plt.subplots(figsize=(7,4))
#fig2, ax2 = plt.subplots()

#with open("../conf/config_birdview.yml", "r") as ymlfile:
#cfg = yaml.load(ymlfile)
width_og, height_og = 540,960
#corner_points = [[1200,10],[1279,719],[100,10],[10,719]]
#corner_points = [[1719,1],[1919,1079],[200,1],[1,1079]]
corner_points = [[860,100],[960,540],[100,100],[1,540]]
#corner_points = [[1919,1],[1919,1079],[1,1],[1,1079]]
img_path = 'bk.png'
size_frame = 960
imgIn = imutils.resize(cv2.imread(img_path), width=int(size_frame))
'''
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]
'''
matrix,imgOutput = compute_perspective_transform(corner_points,width_og,height_og,imgIn)

theight,twidth,_ = imgOutput.shape
blank_image = np.zeros((theight,twidth,3), np.uint8)
theight = blank_image.shape[0]
twidth = blank_image.shape[1] 
dim = (twidth, theight)
# loop over the frames from the video stream
while True:
	count+=1
	px = np.array([])
	py = np.array([])

	# read the next frame from the file
	if config.Thread:
		frame = cap.read()

	else:
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end of the stream
		if not grabbed:
			break
	width, height, inference_time, mask_results = yolo.inference(frame)
        
	# resize the frame and then detect people (and only people) in it
	#frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# initialize the set of indexes that violate the max/min social distance limits
	#serious = set()
	#abnormal = set()
	# loop over the results
	array_groundpoints = []
	# extract all centroids from the results and compute the
	# Euclidean distances between all pairs of the centroids
	for (i, (prob, bbox, centroid)) in enumerate(results):
		'''
		centroids = np.array([r[2] for r in results])
		
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of the centroid pairs
					serious.add(i)
					serious.add(j)
                # update our abnormal set if the centroid distance is below max distance limit
				if (D[i, j] < config.MAX_DISTANCE) and not serious:
					abnormal.add(i)
					abnormal.add(j)
		'''
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = COLOR_GREY
		array_groundpoints.append(centroid)
		# if the index pair exists within the violation/abnormal sets, then update the color
		'''
		if i in serious:
			color = COLOR_RED
		elif i in abnormal:
			color = COLOR_YELLOW #orange = (0, 165, 255)
		'''
		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 2)
		cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
		cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
		cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
		cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
		px = np.append(px,cX)
		py = np.append(py,cY)

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	nomask_counter = 0
	b_serious = set()
	b_abnormal = set()
	if len(results) >= 2:
		for detection in mask_results:
			id, name, confidence, x, y, w, h = detection
			#cx = x + (w / 2)
			#cy = y + (h / 2)
			color = colors[id]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "%s (%s)" % (name, round(confidence, 2))
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
			if id == 2 or id == 1:
				nomask_counter += 1
				print(nomask_counter)
##########################BIRD VIEW POINTS TRANS################################
		#data_arr = []
		transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
		#img = cv2.imread("bk.png")
		#img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
		bird_view_img = cv2.resize(imgOutput, dim, interpolation = cv2.INTER_AREA)
		bX = []
		bY = []
		for point in transformed_downoids:
			x,y = point
			#x,y = abs(x),abs(y)
			#x,y = round(x/2+300),round(y/2+300)
			bX.append(y)
			bY.append(x)

			centroids = np.array(list(zip(bX, bY)))
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number of pixels
					if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of the centroid pairs
						b_serious.add(i)
						b_serious.add(j)
					# update our abnormal set if the centroid distance is below max distance limit
					if (D[i, j] < config.MAX_DISTANCE) and not b_serious:
						b_abnormal.add(i)
						b_abnormal.add(j)
			b_color = COLOR_GREY
			if i in b_serious:
				b_color = COLOR_RED
			elif i in b_abnormal:
				b_color = COLOR_YELLOW

			cv2.circle(bird_view_img, (x,y), BIG_CIRCLE, b_color, 2)
			cv2.circle(bird_view_img, (x,y), SMALL_CIRCLE, b_color, -1)
################################################################################
	data_arr = []
	#pxy = list(zip(px.astype(int), py.astype(int)))
	#print(pxy)
	arrx = px.astype(int)
	arrx = arrx.astype(str)
	arry = py.astype(int)
	arry = arry.astype(str)
	arrx = ','.join(arrx.tolist())
	arry = ','.join(arry.tolist())
	
	social_distance = '0'
	mask = nomask_counter
################################ALERT###########################################	
	if len(b_serious) >= config.Threshold:
		#cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
		if config.ALERT:
			print("[警告]未保持安全社交距離")
			social_distance = len(b_serious)+len(b_abnormal)
			print("未保持安全社交距離人數 : "+str(social_distance))
	#mydata = (('x',int(i[0])),('y',int(i[1])))
	mydata = {'scene':'scene1', 'x': arrx, 'y': arry, 'social_distance': social_distance, 'mask': mask }
	#data_arr.append(mydata)s
	#print(mydata)
##########################SEND DATA TO BACKEND API##############################
	if count%30==0 and config.API==True:
		r = requests.post(config.API_URL, data = mydata)
		print("[提示] 資料傳送")
		print(r.content)
		count = 2

	# draw some of the parameters
	Safe_Distance = "Social Distance: >{} px".format(config.MAX_DISTANCE)
	cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
	#Threshold = "Threshold: {}".format(config.Threshold)
	#cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

    # draw the total number of social distancing violations on the output frame
	text = "Under Safe Distance: {}".format(len(b_serious))
	cv2.putText(frame, text, (10, frame.shape[0] - 55),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

	text1 = "Concentrated: {}".format(len(b_abnormal))
	cv2.putText(frame, text1, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
	#if len(serious)>2 or len(abnormal)>2:
	if len(px)>2 or len(py)>2:
		if count>1:
			ax1.cla()
		plt.ion()
		k = gaussian_kde(np.vstack([bX, bY]))
		xi, yi = np.mgrid[0:frame.shape[1]:72*1j,0:frame.shape[0]:128*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		ax1.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
		ax1.set_xlim(0, frame.shape[1])
		ax1.set_ylim(0, frame.shape[0])

		plt.gca().invert_yaxis()
		plt.gca().invert_xaxis()
		plt.show(block=False)
		del px, py
	"""
	if len(px)>2 or len(py)>2:
		if count>1:
			ax1.cla()
		plt.ion()
		k = gaussian_kde(np.vstack([px, py]))
		xi, yi = np.mgrid[0:frame.shape[1]:72*1j,0:frame.shape[0]:128*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		ax1.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
		ax1.set_xlim(0, frame.shape[1])
		ax1.set_ylim(0, frame.shape[0])

		plt.gca().invert_yaxis()
		plt.show(block=False)
		del px, py
	"""

	if args["display"] > 0:
		#cv2.imshow("Bird view", cv2.flip(bird_view_img,-1))
		cv2.namedWindow('Bird View',cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Bird View",720,405)
		try:
			cv2.imshow("Bird View", cv2.rotate(bird_view_img, cv2.ROTATE_90_CLOCKWISE))
		except:
			print('nothing detected')

		cv2.namedWindow('Realtime Vision',cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Realtime Vision",720,405)
		cv2.imshow("Realtime Vision", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
    # update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("===========================")
print("[提示] 總時長: {:.2f}".format(fps.elapsed()))
print("[提示] 估幀數: {:.2f}".format(fps.fps()))

# close any open windows
cv2.destroyAllWindows()