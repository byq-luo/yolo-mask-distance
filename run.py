from mylib import config, thread
from mylib.yolo import YOLO
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, cv2, os, time , imutils ,schedule
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import requests

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="mylib/videos/street01.mp4",
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
yolo = YOLO("models/yolo-fastest.cfg", "models/yolo-fastest.weights", classes)
#yolo.size = int(args.size)
yolo.size = 416
#yolo.confidence = float(args.confidence)
yolo.confidence = 0.5
colors = [(50, 255, 50), (50, 165, 255), (50, 50, 255)]
count = 0

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("")
	print("[INFO] Looking for GPU")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the camera
if not args.get("input", False):
	print("[INFO] Starting the live stream..")
	vs = cv2.VideoCapture(config.url)
	if config.Thread:
			cap = thread.ThreadingClass(config.url)
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] Starting the video..")
	vs = cv2.VideoCapture(args["input"])
	if config.Thread:
			cap = thread.ThreadingClass(args["input"])

# start the FPS counter
fps = FPS().start()
fig, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()

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
	serious = set()
	abnormal = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		for detection in mask_results:
			id, name, confidence, x, y, w, h = detection
			#cx = x + (w / 2)
			#cy = y + (h / 2)
			color = colors[id]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "%s (%s)" % (name, round(confidence, 2))
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
		
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
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

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (200, 200, 200)
		
		# if the index pair exists within the violation/abnormal sets, then update the color
		
		if i in serious:
			color = (0, 165, 255)
		elif i in abnormal:
			color = (0, 255, 255) #orange = (0, 165, 255)
		

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 2)
		px = np.append(px,cX)
		py = np.append(py,cY)

	data_arr = []
	#pxy = list(zip(px.astype(int), py.astype(int)))
	#print(pxy)
	arrx = px.astype(int)
	arrx = arrx.astype(str)
	arry = py.astype(int)
	arry = arry.astype(str)
	arrx = ','.join(arrx.tolist())
	arry = ','.join(arry.tolist())
	#mydata = (('x',int(i[0])),('y',int(i[1])))
	mydata = {'scene':'scene1','x': arrx, 'y': arry}
	#data_arr.append(mydata)
	#print(mydata)
	#api = False
	if count%30==0 and config.API==True:
		r = requests.post(config.API_URL, data = mydata)
		print("[INFO] Data Send")
		print(r.content)
		count = 2
	#data_arr = []

	# draw some of the parameters
	Safe_Distance = "Social Distance Set: >{} px".format(config.MAX_DISTANCE)
	cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
	Threshold = "Threshold: {}".format(config.Threshold)
	cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

    # draw the total number of social distancing violations on the output frame
	text = "Under Safe Distance: {}".format(len(serious))
	cv2.putText(frame, text, (10, frame.shape[0] - 55),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

	text1 = "Concentrated: {}".format(len(abnormal))
	cv2.putText(frame, text1, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
	#if len(serious)>2 or len(abnormal)>2:
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

	if args["display"] > 0:
		# show the output frame
		cv2.imshow("即時社交距離偵測", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
    # update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("===========================")
print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# close any open windows
cv2.destroyAllWindows()