# import the necessary packages
from imutils.video import VideoStream
from flask import Response,Flask,render_template,request,redirect
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import threading
import argparse
import datetime
import imutils
import time
import cv2
import sqlite3
import os
import socket
import sys
import logging as log
# from gevent.pywsgi import WSGIServer
from centroidtracker import CentroidTracker
#TFLite
from tensorflow.lite.python.interpreter import Interpreter
#movidius
try:
	from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
	from openvino.inference_engine import IENetwork, IEPlugin
from line_notify import LineNotify
import numpy as np
##########################################################
conn = sqlite3.connect('/home/pi/smartcam/smart_cam.db',check_same_thread=False)
print ("Opened database successfully")
##########################################################
##########################Global variable################################
# os.system("export DISPLAY=:0")
outputFrame = None
wifi_status = 0
wifi_name_now =""
line_alert_time = 0
#first_time load database
first_con = conn.cursor()
first_con.execute("SELECT * FROM smart_setting")
_setting = first_con.fetchall()
first_con.execute("SELECT * FROM select_class")
_class = first_con.fetchall()
_reset = 0
#########################################################
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__,template_folder="template")

########################################################################################
@app.route("/")
def index():
	# return the rendered template
	return render_template("body/home.html")
##########################################################################################################
@app.route('/report',)
def report():
	cur = conn.cursor()
	cur.execute("SELECT * FROM smart_report ORDER BY id DESC")
	report_select = cur.fetchall()
	#print(report_select)
	return render_template('body/report.html',report_select=report_select)
##########################################################################################################
@app.route('/setting',methods=['GET', 'POST'])
def setting():
	if request.method == "POST":
		################################################
		################################################
		cam_set = request.form['cam_set']
		Rtsp_text = request.form['Rtsp_text']
		cam_res = request.form['cam_res']
		cam_acc = request.form['cam_acc']
		line_alert = request.form['line_alert']
		try:
			cnn = request.form['cnn']
		except:
			cnn = str(0)
		conn.execute("UPDATE smart_setting SET camera_set = '"+cam_set+"',rtsp = '"+Rtsp_text+"',camera_acc = '"+cam_acc+"',camera_res = '"+cam_res+"',Line = '"+line_alert+"',cnn = '"+cnn+"'")
		conn.commit()
		########################################
		try:
			person = request.form['person']
		except:
			person = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+person+"' Where class = '%s' "%('person'))
		conn.commit()
		try:
			cat = request.form['cat']
		except:
			cat = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+cat+"' Where class = '%s' "%('cat'))
		conn.commit()
		try:
			dog = request.form['dog']
		except:
			dog = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+dog+"' Where class = '%s' "%('dog'))
		conn.commit()
		try:
			brid = request.form['brid']
		except:
			brid = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+brid+"' Where class = '%s' "%('brid'))
		conn.commit()
		try:
			car = request.form['car']
		except:
			car = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+car+"' Where class = '%s' "%('car'))
		conn.commit()
		try:
			motorcycle = request.form['motorcycle']
		except:
			motorcycle = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+motorcycle+"' Where class = '%s' "%('motorcycle'))
		conn.commit()
		try:	
			bicycle = request.form['bicycle']
		except:
			bicycle = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+bicycle+"' Where class = '%s' "%('bicycle'))
		conn.commit()
		try:
			truck = request.form['truck']
		except:
			truck = str(0)
		conn.execute("UPDATE select_class SET select_classname='"+truck+"' Where class = '%s' "%('truck'))
		conn.commit()
		###############SELECT CLASS######################
		# print(request.form['st_time'],request.form['en_time'])
		###############DayWork###########################
		try:
			mon = request.form['Monday']
		except:
			mon = str(0)
		try:
			tue = request.form['Tuesday']
		except:
			tue = str(0)
		try:
			wed = request.form['Wednesday']
		except:
			wed = str(0)
		try:
			thu = request.form['Thursday']
		except:
			thu = str(0)
		try:
			fri = request.form['Friday']
		except:
			fri = str(0)
		try:
			sat = request.form['Saturday']
		except:
			sat = str(0)
		try:
			sun = request.form['Sunday']
		except:
			sun = str(0)
		try:
			alltime = request.form['alltime']
		except:
			alltime = str(0)
		try:
			st_time = request.form['st_time']
		except:
			st_time = str(0)
		try:
			en_time = request.form['en_time']
		except:
			en_time = str(0)								
		conn.execute("UPDATE smart_daywork SET monday='"+mon+"' , tuesday='"+tue+"',wednesday='"+wed+"',\
		thursday='"+thu+"',friday='"+fri+"',saturday='"+sat+"',sunday='"+sun+"',st_time='"+st_time+"',en_time='"+en_time+"',alltime='"+alltime+"' ")
		conn.commit()
		#################################################
		#print(person,cat,dog,brid,car,motorcycle,bicycle,bicycle,truck)
		# if cnn_check[0] != cnn:
		# os.system("sudo reboot")
		return redirect(request.referrer)
	else:
		################################################
		cur = conn.cursor()
		cur.execute("SELECT * FROM smart_setting") 
		setting_value = cur.fetchall()
		################################################
		cur.execute("SELECT * FROM select_class") 
		select = cur.fetchall()
		select_class = []
		for i in range(len(select)):
			if select[i][2] == 1:
				select_class.append('checked')
			else:
				select_class.append('0')
		################################################
		cur.execute("SELECT * FROM smart_daywork")
		daywork = cur.fetchall() #column ที่ 0 คือ id
		allday_check = 0
		daywork_check = []
		for i in range(1,11):
			if i == 10 :
				
				if daywork[0][i]:
					daywork_check.append("checked")
					alltime = 1
				else:
					daywork_check.append(" ")
					alltime = 0
			elif i < 8 :
				if daywork[0][i]:
					daywork_check.append("checked")
					allday_check += 1
				else:
					daywork_check.append(" ")
			else:
				daywork_check.append(daywork[0][i])
		#เช็คว่าให้แจ้งเตือนทุกวันหรือไม่ ถ้าใช่จะทำการ checked หน้า Allday
		if allday_check >= 7:
			daywork_check.append("checked")
			allday = 1
		else:
			daywork_check.append(" ")
			allday = 0
		# print(daywork)
		# print(daywork_check)
		################################################
		if setting_value[0][6] == 1:
			cnn = 'checked'
		else:
			cnn = ''
		# print(setting_value)
		#print(select_class)
		return render_template('body/settings.html',setting_value=setting_value,select_class=select_class,Cnn=cnn,daywork_check=daywork_check,alltime=alltime,allday=allday)

##########################################################################################################
def insert_wifi(ssid,password):
	t1 = time.time()
	t2 = time.time()
	while (t1 < t2 + 10):
		t1 = time.time()
		con_check = os.popen('sudo nmcli device wifi con \"'+ ssid +'\" password \"'+ password +'\"').read()
		print(con_check)
		if con_check.find('successfully') != -1:
			break
		elif con_check.find('failed') != -1:
			os.system('sudo nmcli connection delete id '+ ssid)
			# os.system('sudo nmcli con up Hostspot')
		#os.system('sudo nmcli device wifi con \"'+ ssid +'\" password \"'+ password +'\"')
@app.route('/setwifi')
def setwifi():
	global wifi_status ,wifi_name_now,status
	try:
		gw = os.popen("ip -4 route show default").read().split()
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect((gw[2], 0))
		ipaddr = s.getsockname()[0]
		gateway = gw[2]
		host = socket.gethostname()
		#print ("IP:", ipaddr, " GW:", gateway, " Host:", host)
	except:
		ipaddr = '0.0.0.0'
	return render_template('body/setwifi.html',wifi_status=wifi_status ,wifi_name_now=wifi_name_now,wifi_ip=ipaddr)

##########################################################################################################
def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir',default='models')
    parser.add_argument('--graph',default='detect.tflite')
    parser.add_argument('--labelsTF',default='labelmap.txt')
    parser.add_argument('--resolution', default='640x480')
    parser.add_argument("-m", "--model", type=str,default='/home/pi/smartcam/models/movidius/ssd_mobilenet_v2_coco.frozen.xml')
    parser.add_argument("-l", "--cpu_extension",type=str,default=None)
    parser.add_argument("-pp", "--plugin_dir", type=str, default=None)
    parser.add_argument("-d", "--device",default="MYRIAD", type=str)
    parser.add_argument("--labels", default='/home/pi/smartcam/models/movidius/coco.names.txt', type=str)
    return parser.parse_args()
##########################################################################################################
def display_show(rects,t1_image,t2_image,ckname,frame_q,objects_first,notify,ct):
	global line_alert_time
	ckid = ''
	text=''
	timea=10
	objects = ct.update(rects)
	cur = conn.cursor()
	for (objectID, centroid) in objects.items():
		text = "{}".format(objectID)
		ckid = ckid + text + ','
		cv2.putText(frame_q, 'ID '+text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (123, 244, 80), 2)
		cv2.circle(frame_q, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	############################
	#Inset Database
	if len(objects) > 0 and ckname != '':
		if (int(t1_image-t2_image) >= timea or objects_first == 0) and rects != None and line_alert_time == 1 :
			#ลบ , ตัวสุดท้ายออก
			ckname = ckname[:-1]
			ckid = ckid[:-1]
			##บันทึกรูปภาพ
			img_name = 'img'+datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')+'.jpg'
			img_database = './screenshots/'+img_name
			cv2.imwrite('/home/pi/smartcam/static/screenshots/'+img_name,frame_q)
			time_insert = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
			#Send Line
			try:
				notify.send('ID '+ckid+' '+ckname+' Detected',image_path='/home/pi/smartcam/static/screenshots/'+img_name)
				Line_Send = 1
				log.critical("Line notify Done")
			except :
				Line_Send = 0
				log.error("Line notify Error")
			##อัพข้อมูลหน้า report ลง ดาต้าเบส
			cur.execute("INSERT INTO smart_report (id_class,Filename,Class_detect,Line,Time) VALUES ('%s','%s','%s','%s','%s')"%(ckid,img_database,ckname,Line_Send,time_insert))
			conn.commit()
			##Reset
			t2_image =  time.time()
			#Inset Print
			log.critical("Inset Done")
			objects_first = 1
	else:
		if int(t1_image-t2_image) >= 2 and int(t1_image-t2_image) <= 4:
			None
		else:
			t2_image =  time.time()
			objects_first = 0
	return frame_q,objects_first,t2_image
##########################################################################################################
def select_option():
	#SettingCam
	cur = conn.cursor()
	cur.execute("SELECT * FROM smart_setting") 
	setting_value = cur.fetchall()
	#print(setting_value)
	cam_select = setting_value[0][1]
	cam_rtsp = setting_value[0][2]
	min_conf_threshold = setting_value[0][3]*0.01
	#Line notify
	ACCESS_TOKEN = str(setting_value[0][5])
	notify = LineNotify(ACCESS_TOKEN)

	#Select List detect
	cur.execute("SELECT * FROM select_class")
	myresult = cur.fetchall()
	#print(myresult[0][2])
	detect_list = []
	for i in range(len(myresult)):
		if myresult[i][2] == 1 :
			detect_list.append(myresult[i][1].lower())
	if setting_value[0][4] == '1080': resW,resH = 1920,1080
	elif setting_value[0][4] == '720': resW,resH = 1280,720
	elif setting_value[0][4] == '480': resW,resH = 640,480
	elif setting_value[0][4] == '360': resW,resH = 480,360
	
	return cam_select,cam_rtsp,min_conf_threshold,notify,detect_list,resW,resH
##########################################################################################################
def movidius(frameCount):
	global outputFrame, lock,_reset
	ct = CentroidTracker()
	args = argsparser()
	#ใช้ Log แทน Print
	log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
	#Funtion Select
	cam_select,cam_rtsp,min_conf_threshold,notify,detect_list,resW,resH = select_option()
	# warmup
	if cam_select == 0:
		vs = cv2.VideoCapture(0)
	else:
		vs = cv2.VideoCapture(cam_rtsp)
	imW, imH 	 	= int(resW), int(resH)
	vs.set(3,imW)
	vs.set(4,imH)
	time.sleep(0.2)
	##################
	#โหลดโมดูล
	model_xml = args.model
	model_bin = os.path.splitext(model_xml)[0] + ".bin"
	log.info("Initializing plugin for {} device...".format(args.device))
	plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
	if args.cpu_extension and 'CPU' in args.device:
		plugin.add_cpu_extension('args.cpu_extension')
	# Read IR
	log.info("Reading IR...")
	net    = IENetwork(model=model_xml, weights=model_bin)
	if plugin.device == "CPU":
		supported_layers = plugin.get_supported_layers(net)
		not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
		if len(not_supported_layers) != 0:
			log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
						format(plugin.device, ', '.join(not_supported_layers)))
			log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
						"or --cpu_extension command line argument")
			sys.exit(1)
	assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
	assert len(net.outputs) == 1, "Demo supports only single output topologies"
	input_blob = next(iter(net.inputs))
	out_blob   = next(iter(net.outputs))
	log.info("Loading IR to the plugin...")
	exec_net   = plugin.load(network=net, num_requests=2)
	# Read and pre-process input image
	n, c, h, w = net.inputs[input_blob].shape
	del net
	if args.labels:
		with open(args.labels, 'r') as f:
			labels_map = [x.strip() for x in f]
	else:
		labels_map = None

	cur_request_id  = 0
	next_request_id = 1

	log.info("Starting inference in async mode...")
	log.info("To switch between sync and async modes press Tab button")
	log.info("To stop the demo execution press Esc button")
	is_async_mode = False
	##################
	# Initialize frame rate calculation
	frame_rate_calc = 1
	freq = cv2.getTickFrequency()
	font = cv2.FONT_HERSHEY_SIMPLEX
	##########################################
	objects_first = 0
	t1_image =  time.time()
	t2_image =  time.time()
	##########################################
	log.info("Detect On")
	while True:
		t1 = cv2.getTickCount()
		#ชื่อคลาสและไอดีของคลาส
		ckname = ''
		object_name = None
		ret,frame_q = vs.read()

		if is_async_mode:
			in_frame = cv2.resize(frame_q, (w, h))
			in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
			in_frame = in_frame.reshape((n, c, h, w))
			exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
		else:
			in_frame = cv2.resize(frame_q, (w, h))
			in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
			in_frame = in_frame.reshape((n, c, h, w))
			exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
		rects = []
		if exec_net.requests[cur_request_id].wait(-1) == 0:
			# Parse detection results of the current request
			res = exec_net.requests[cur_request_id].outputs[out_blob]
			for obj in res[0][0]:
				# Draw only objects when probability more than specified threshold
				if obj[2] > min_conf_threshold:
					# Draw label
					class_id = int(obj[1])
					object_name = labels_map[class_id] if labels_map else str(class_id)
					if object_name in detect_list:
						ckname = ckname + object_name + ','
						xmin = int(obj[3] * imW)
						ymin = int(obj[4] * imH)
						xmax = int(obj[5] * imW)
						ymax = int(obj[6] * imH)
						cv2.rectangle(frame_q, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
						label = '%s: %s%%' % (object_name, int(round(obj[2] * 100, 1)))
						#print(label)
						labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
						label_ymin = max(ymin, labelSize[1] + 10) 
						color = color_name(object_name)
						cv2.rectangle(frame_q, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
						cv2.putText(frame_q, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
						x = np.array([xmin, ymin, xmax, ymax])
						# print(x)
						rects.append(x.astype("int"))
		cv2.putText(frame_q,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
		
		#Funtion display_show
		t1_image = time.time()
		frame_q,objects_first,t2_image = display_show(rects,t1_image,t2_image,ckname,frame_q,objects_first,notify,ct)

		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc = 1/time1

		if is_async_mode:
			cur_request_id, next_request_id = next_request_id, cur_request_id
		with lock:
			outputFrame = frame_q.copy()
		if _reset == 1:
			sys.exit()
	vs.stop()

def TFlite(frameCount):
	# grab global references to the  output frame, and # lock variables
	global outputFrame, lock,_reset
	ct = CentroidTracker()
	#Funtion Select
	cam_select,cam_rtsp,min_conf_threshold,notify,detect_list,resW,resH = select_option()
	# warmup
	if cam_select == 0:
		vs = cv2.VideoCapture(0)
	else:
		vs = cv2.VideoCapture(cam_rtsp)
	imW, imH 	 	= int(resW), int(resH)
	vs.set(3,imW)
	vs.set(4,imH)
	time.sleep(2.0)
	##################
	args 			= argsparser()
	MODEL_NAME 		= args.modeldir
	GRAPH_NAME 		= args.graph
	LABELMAP_NAME 	= args.labelsTF

	#######################
	# min_conf_threshold = args.threshold
	#######################
	CWD_PATH 	 	= os.getcwd()
	# Path to .tflite file, which contains the model that is used for object detection
	PATH_TO_CKPT 	= os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
	# Path to label map file
	PATH_TO_LABELS 	= os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
	# Load the label map
	with open(PATH_TO_LABELS, 'r') as f:
		labels = [line.strip() for line in f.readlines()]
	if labels[0] == '???':
		del(labels[0])

	# Load the Tensorflow Lite model and get details
	interpreter = Interpreter(model_path=PATH_TO_CKPT)
	interpreter.allocate_tensors()

	input_details 	= interpreter.get_input_details()
	output_details 	= interpreter.get_output_details()
	height 			= input_details[0]['shape'][1]
	width 			= input_details[0]['shape'][2]

	floating_model 	= (input_details[0]['dtype'] == np.float32)
	input_mean 		= 127.5
	input_std 		= 127.5

	# Initialize frame rate calculation
	frame_rate_calc = 1
	freq = cv2.getTickFrequency()
	font = cv2.FONT_HERSHEY_SIMPLEX
	##########################################
	t1_image =  time.time()
	t2_image =  time.time()
	objects_first = 0
	##########################################
	log.info("Detect On")
	while True:
		t1 = cv2.getTickCount()
		ret,frame_q = vs.read()

		frame_resized 	= cv2.resize(frame_q, (width, height))
		input_data 		= np.expand_dims(frame_resized, axis=0)

		# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
		if floating_model:
			input_data = (np.float32(input_data) - input_mean) / input_std

		# Perform the actual detection by running the model with the image as input
		interpreter.set_tensor(input_details[0]['index'],input_data)
		interpreter.invoke()

		# Retrieve detection results
		boxes 	= interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
		classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
		scores 	= interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
		#num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
		ckname = ''
		object_name = None
		rects = []
		# Loop over all detections and draw detection box if confidence is above minimum threshold
		for i in range(len(scores)):
			if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
					# Draw label
					object_name = labels[int(classes[i])]
					if object_name in detect_list:
						ckname = ckname + object_name + ','
						ymin = int(max(1,(boxes[i][0] * imH)))
						xmin = int(max(1,(boxes[i][1] * imW)))
						ymax = int(min(imH,(boxes[i][2] * imH)))
						xmax = int(min(imW,(boxes[i][3] * imW)))
						cv2.rectangle(frame_q, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
						label = '%s: %d%%' % (object_name, int(scores[i]*100))
						#print(label)
						labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
						label_ymin = max(ymin, labelSize[1] + 10) 
						color = color_name(object_name)
						cv2.rectangle(frame_q, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+45, label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
						cv2.putText(frame_q, label, (xmin+45, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
						x = np.array([xmin, ymin, xmax, ymax])
						# print(x)
						rects.append(x.astype("int"))

		cv2.putText(frame_q,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

		#Funtion display_show
		t1_image = time.time()
		frame_q,objects_first,t2_image = display_show(rects,t1_image,t2_image,ckname,frame_q,objects_first,notify,ct)

		with lock:
			outputFrame = frame_q.copy()
		if _reset == 1:
			sys.exit()
		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc = 1/time1
	vs.stop()
##########################################################################################################
def color_name(object_name):
	if object_name == 'person':
		color = (0,255,0)
	elif object_name == 'cat':
		color = (255,0,255)
	elif object_name == 'dog':
		color = (255,255,0)
	elif object_name == 'brid':
		color = (255, 146, 0)
	elif object_name == 'car':
		color = (255, 253, 5)
	elif object_name == 'motorcycle':
		color = (254, 165, 198)
	elif object_name == 'bicycle':
		color = (254, 255, 198)
	elif object_name == 'truck':
		color = (124, 118, 0)
	else:
		color = (0, 0, 185)
	return color
##########################################################################################################	
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")
########################################################################
def check_sql_onload():
	global _reset,_setting,_class,line_alert_time
	cur = conn.cursor()
	cur.execute("SELECT * FROM smart_setting")
	setting_setting = cur.fetchall()

	cur.execute("SELECT * FROM select_class")
	setting_class = cur.fetchall()
	################################################################
	#####################ตรวจสอบการแจ้งเตือนบน Line วัน/เวลา#######################
	x = datetime.datetime.now()
	x = x.strftime("%A")
	cur.execute("SELECT * FROM smart_daywork")
	main_daywork = cur.fetchall()
	_daycheck = day_check(main_daywork,x)
	# print(setting_cnn[0])
	_timenow = datetime.datetime.now()
	# _timenow = _timenow.strftime("%I:%M %p")
	_timenow = _timenow.strftime("%H:%M")
	# print(_timenow) # เวลาปัจจุบัน
	cur.execute("SELECT * FROM smart_daywork")
	_timedatabase = cur.fetchall()
	if _timedatabase[0][10] != 1:
		st_time = _timedatabase[0][8] #เวลาเริ่มต้นที่ให้ทำงาน
		st_time = datetime.datetime.strptime(st_time, "%I:%M %p")
		st_time = datetime.datetime.strftime(st_time, "%H:%M")
		en_time = _timedatabase[0][9] #เวลาสิ้นหลุดที่ให้ทำงาน
		en_time = datetime.datetime.strptime(en_time, "%I:%M %p")
		en_time = datetime.datetime.strftime(en_time, "%H:%M")
	# print(st_time , en_time)
	# ถ้า 0 คือวันนี้แจ้งเตือน 1 คือไม่แจ้งเตือน
	if _daycheck == 0 :
		# ถ้าวันนี้ ติ้ก AllTime ไม่ต้องเช็คเวลา
		if _timedatabase[0][10] == 1 :
			line_alert_time = 1
		else:
			# ถ้าไม่มีการติ้ก Alltime ให้เช็คจากเวลา
			# ถ้า เวลาปัจจุบันน้อยกว่าเวลาที่ตั้งค่า
			if _timenow < st_time and _timenow > en_time :
				line_alert_time = 1
			else:
				line_alert_time = 0
	else:
		line_alert_time = 0
	# ถ้า เวลาปัจจุบันมากกว่า เวลาที่ให้เริ่มทำงาน
	################################################################
	if( _setting == setting_setting) and ( _class == setting_class):
		_reset = 0
	else:
		_reset = 1
		if setting_setting[0][6] == 0:
			t = threading.Thread(target=TFlite, args=(32,))
			t.daemon = True
			t.start()
		else:
			t = threading.Thread(target=movidius, args=(32,))
			t.daemon = True
			t.start()
		_setting = setting_setting
		_class = setting_class
	#print(_reset)

def day_check(main_daywork,xday):
    day_reture = 0
    if xday == "Monday":
        if main_daywork[0][1] == 0 : None
        else: day_reture = 1
    elif xday == "Tuesday":
        if main_daywork[0][2] == 0 : None
        else: day_reture = 1
    elif xday == "Wednesday":
        if main_daywork[0][3] == 0 : None
        else: day_reture = 1
    elif xday == "Thursday":
        if main_daywork[0][4] == 0 : None
        else: day_reture = 1
    elif xday == "Friday":
        if main_daywork[0][5] == 0 : None
        else: day_reture = 1
    elif xday == "Saturday":
        if main_daywork[0][6] == 0 : None
        else: day_reture = 1
    elif xday == "Sunday":
        if main_daywork[0][7] == 0 : None
        else: day_reture = 1
    return day_reture
def main():
	global wifi_name_now,wifi_status
	status = os.popen('iwconfig wlan0').read()
	#print(status)
	if status.find("ESSID:off/any") != 23 and status.find("limit:7") != 77:
		wifi_status = 1
		wifi_name_local = os.popen('iwgetid').read().split('"')
		wifi_name_now = wifi_name_local[1]
	else:
		wifi_status = 0
		wifi_name_now = "None"

	cur = conn.cursor()
	cur.execute("SELECT cnn FROM smart_setting") 
	setting_cnn = cur.fetchone()
	#ตรวจสอบ CNN ที่ใช้
	if setting_cnn[0] == 0:
		t = threading.Thread(target=TFlite, args=(32,))
		t.daemon = True
		t.start()
	else:
		t = threading.Thread(target=movidius, args=(32,))
		t.daemon = True
		t.start()

	# start the flask app
	app.run(host='0.0.0.0', port='8000', debug=True,threaded=True, use_reloader=False)
# check to see if this is the main thread of execution
if __name__ == '__main__':
	main()
scheduler = BackgroundScheduler()
scheduler.add_job(func=check_sql_onload, trigger="interval", seconds=2)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())