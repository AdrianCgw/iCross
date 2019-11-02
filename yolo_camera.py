# USAGE
# python yolo_camera.py --yolo yolo-coco -v 1 -s 4 -i 9 -c 0.2

'''
We are searching for 'traffic light' which is the id 9

'''
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
# INSTALL need opencv 3.4.2
# conda install opencv-contrib-python
import os
from playsound import playsound
# INSTALL: pip install playsound
import threading
from timeit import default_timer as timer

if True:
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()    
    '''
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image")
    '''
    ap.add_argument('-s','--scaledown', type=int, default = 1, help = 'scaledown factor for faster processing')
    ap.add_argument('-v','--video', type=int, default = 0, help = 'videcam source')
    ap.add_argument("-y", "--yolo", required=True,
    	help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
    	help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
    	help="threshold when applyong non-maxima suppression")
    ap.add_argument("-i", "--object_id", type=int, default=None,
        help='look for a single object type')
    args = vars(ap.parse_args())

    print('Using the following args:')
    print(args['yolo'])
    print(args['confidence'])
    print(args['threshold'])

def yolo_classify(image, wanted_id = None):
    
    # we get the image from the computer screen:
    if False:        
        # load our input image and grab its spatial dimensions
        image = cv2.imread(args["image"])

    (H, W) = image.shape[:2]
    
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
      # loop over each of the detections
      for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        if not wanted_id:
            classID = np.argmax(scores)
            confidence = scores[classID]
        else:
            classID = wanted_id
            confidence = scores[classID]
            
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
          # scale the bounding box coordinates back relative to the
          # size of the image, keeping in mind that YOLO actually
          # returns the center (x, y)-coordinates of the bounding
          # box followed by the boxes' width and height
          box = detection[0:4] * np.array([W, H, W, H])
          (centerX, centerY, width, height) = box.astype("int")
          
          # use the center (x, y)-coordinates to derive the top and
          # and left corner of the bounding box
          x = int(centerX - (width / 2))
          y = int(centerY - (height / 2))
          
          # update our list of bounding box coordinates, confidences,
          # and class IDs
          boxes.append([x, y, int(width), int(height)])
          confidences.append(float(confidence))
          classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    
    # return also the top confidence object:
    top = None
    if len(idxs) > 0:
        tid = np.argmax(confidences)
        (x, y) = (boxes[tid][0], boxes[tid][1])
        (w, h) = (boxes[tid][2], boxes[tid][3])
        top = {'confidence':confidences[tid],'x':x, 'y':y, 'w':w, 'h':h}

    # ensure at least one detection exists
    if len(idxs) > 0:
    	# loop over the indexes we are keeping
    	for i in idxs.flatten():
    		# extract the bounding box coordinates
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])
    
    		# draw a bounding box rectangle and label on the image
    		color = [int(c) for c in COLORS[classIDs[i]]]
    		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, color, 2)
    
    # we display the image in the main loop
    if False:
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    else:
        return image, top
 
# Camera object #################################################
#Encapsulate the robot webcam IO:
class CameraIO ():
    def __init__(self, port = 0):
        self.cam = cv2.VideoCapture(port)

    #Returns ret_val,image
    def read(self):
        return self.cam.read()
    
    def __del__(self):
        self.cam.release()        

# %%
        
def get_direction(surface, x_cam, y_cam):
    ''' Returns the direction in which the object is located:
        2 = object very close, you have reached
        3 = object very far or no object detected
        0 = left
        1 = right
        4 = front
    '''
    if not surface or surface < min_surf:
        direct = 3
        deg = 0
        return direct, deg
    if surface > max_surf:
        direct = 2
        deg = 0
        return direct, deg
    zone = int(x_cam * (nb_deg * 2 + 2) / 640)
    if zone < nb_deg:
       #lefts
       direct = 0
       deg = nb_deg -zone
    elif zone > nb_deg +1:
        #right
        direct = 1
        deg = zone - nb_deg - 1
    else:
        # forward
        direct = 4
        deg = 0
    return direct, deg

def soundplay(d,deg, first_pan):    
    if d == 0:
        # left
        for i in range(deg):
            playsound(sound_dir +'left.mp3')
    elif d == 1:
        # right
        for i in range(deg):
            playsound(sound_dir +'right.mp3')
    elif d == 2:
        # reached
        playsound(sound_dir +'reached.mp3')
    elif d == 4:
        # forward
        playsound(sound_dir + 'front.mp3')
    elif d == 3:
        # no detect
        #plays a longer message the first time
        if first_pan:
            playsound(sound_dir + 'pleasepan.mp3')
        else:
            playsound(sound_dir + 'pan.mp3')

# %%

# Main loop ###############################################################
min_surf = 0
max_surf = 51000
nb_deg = 1 
sound_dir = 'sound/'        
play_sound = True
# how much to wait between sound comands
sound_delay = 2.5
# how much time to remember the target's last location
target_memory = 4.5

#args = {'yolo':'yolo_coco/', 'confidence':0.5, 'threshold': 0.3} 
   
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
    
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print(weightsPath, configPath)

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


camera_port = args['video']
#camera = cv2.VideoCapture(camera_port)
camera = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)
# Check if the webcam is opened correctly
if not camera.isOpened():
    raise IOError("Cannot open webcam")
    
wanted_id = args['object_id']
last_sound = timer() - sound_delay
last_target = timer() - target_memory
surface, x_cam , y_cam = None, None, None
#first pan message is longer   
first_pan = True
  
while True:
    return_value, image = camera.read()
    #print("We take a picture of you, check the folder")
    #cv2.imwrite("image.png", image)
    if not return_value:
        continue

    source = image.copy()

    #resize the image
    if args['scaledown'] != 1:
        old_width = image.shape[1]
        old_height = image.shape[0]
        width = int(image.shape[1] / args['scaledown'])
        height = int(image.shape[0] / args['scaledown'])
        dim = (width, height)
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # yolo modifies the image it is given to it    
    img, top = yolo_classify(image,wanted_id) 
    #cv2.imshow('Source', source)

    if args['scaledown'] != 1:
        image = cv2.resize(image, (old_width, old_height), cv2.INTER_LINEAR)
    cv2.imshow('Classification', image)

    # use the object coordinates to direct the user
    if top:
        # we update the last target detection time
        last_target = timer()
        surface = top['w'] * top ['h'] *args['scaledown'] * args['scaledown']
        x_cam = int((top['x'] + top['w'] /2.)* args['scaledown'])
        y_cam = int((top['y'] + top['h'] /2.)* args['scaledown'])
        print(surface, x_cam, y_cam)
    else:
        # remember the last direction recorded
        pass
    
    # we forget old targets:
    if timer() > last_target + target_memory:
        surface = None    
    
    d,deg = get_direction(surface, x_cam, y_cam)
    print(d, deg)
    # play sound only if enough time has passed 
    if timer() > last_sound + sound_delay:
        if play_sound:
            # play sound on a differnt thread
            t = threading.Thread(target = soundplay, args = (d,deg, first_pan))            
            t.setDaemon(True)
            t.start()
            last_sound = timer()
            # set 1st pan to false if direction is 3 (no detection) and wait a bitmore
            if d == 3 and first_pan:
                last_sound = last_sound + 4 #the extra length of the please pan
                first_pan = False
        
    if cv2.waitKey(1) == 27: 
        camera.release()
        cv2.destroyAllWindows()
        break

        