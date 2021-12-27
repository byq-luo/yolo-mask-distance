# base path to YOLO directory
PEOPLE_MODEL_PATH = "yolo"
FACE_MODEL_PATH = "models"
#if backend is sets
API = True
#API_URL = 'http://127.0.0.1:8000/api/plot'
API_URL = 'http://140.128.137.23/%e4%ba%ba%e7%be%a4%e7%9b%a3%e6%8e%a7%e7%b3%bb%e7%b5%b1/python-laravel-backend/public/api/plot'
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#===============================================================================
#=================================\CONFIG./=====================================
""" Below are your desired config. options to set for real-time inference """
# To count the total number of people (True/False).
People_Counter = True
# Threading ON/OFF. Please refer 'mylib>thread.py'.
Thread = False
# Set the threshold value for total violations limit.
Threshold = 2
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video');
# Set url = 0 for webcam.
url = 0
# Turn ON/OFF the email alert feature.
ALERT = True
# Set mail to receive the real-time alerts. E.g., 'xxx@gmail.com'.
#MAIL = ''
# Set if GPU should be used for computations; Otherwise uses the CPU by default.
USE_GPU = True
# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 120
MIN_DISTANCE = 80
#===============================================================================
#===============================================================================
