#python yolo_video.py --yolo yolo-coco

from tkinter import *
from tkinter import ttk,filedialog
from PIL import ImageTk, Image
import cv2
from input_retrieval import *
import numpy as np
import imutils
import time
from scipy import spatial
from polygon import *
from create_mask import *

list_of_vehicles =["auto","bicycle","bus","car","minitruck","motorbike","rickshaw","truck","van"]
FRAMES_BEFORE_CURRENT = 15  
inputWidth, inputHeight = 416, 416
count={}
vehicle_crossed_line_flag = False

#vehicle_count = 0
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
    preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 4),
    dtype="uint8")

def displayVehicleCount(frame, vehicle_count, count):
    try: 
        print('Bicycles: ', count['bicycle'])
    except KeyError:
        count['bicycle'] = 0
    try: 
        print('Motorbikes: ', count['motorbike'])
    except KeyError:
        count['motorbike'] = 0
    try: 
        print('Autorickshaws: ', count['auto'])
    except KeyError:
        count['auto'] = 0
    try: 
        print('Minitrucks: ', count['minitruck'])
    except KeyError:
        count['minitruck'] = 0
    try: 
        print('Cars: ', count['car'])
    except KeyError:
        count['car'] = 0
    try: 
        print('Buses: ', count['bus'])
    except KeyError:
        count['bus'] = 0
    try: 
        print('Rickshaws: ', count['rickshaw'])
    except KeyError:
        count['rickshaw'] = 0
    try: 
        print('Trucks: ', count['truck'])
    except KeyError:
        count['truck'] = 0
    try: 
        print('Vans: ', count['van'])
    except KeyError:
        count['van'] = 0

def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
    x1_line, y1_line, x2_line, y2_line = line_coordinates #Unpacking

    if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and\
        (y_mid_point >= y1_line and y_mid_point <= y2_line+5):
        return True
    return False

def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if(current_time > start_time):
        # os.system('CLS') # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames


def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames, 
#           the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
#         False if the box was not present in the previous frames
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf #Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0: # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height)/2)):
        return False

    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y+ (h//2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count 

                if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                    vehicle_count += 1
                    try:
                                count[LABELS[classIDs[i]]]+=1
                    except KeyError:
                                count[LABELS[classIDs[i]]]=1


                    #vehicle_crossed_line_flag = True
                # else: #ID assigning
                    #Add the current detection mid-point of box to the list of detected items
                # Get the ID corresponding to the current detection

                ID = current_detections.get((centerX, centerY))
                # If there are two detections having the same ID due to being too close, 
                # then assign a new ID to current detection.
                if (list(current_detections.values()).count(ID) > 1):
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1 

                    try:
                                count[LABELS[classIDs[i]]]+=1
                    except KeyError:
                                count[LABELS[classIDs[i]]]=1

                #Display the ID at the center of the box
                cv2.putText(frame, str(ID+1), (centerX, centerY),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 3)
                #f=open("myfile2.txt","w")
                #f.write(str(vehicle_count))

    return vehicle_count, current_detections, count

# def get_frame():
#     if cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             # Return a boolean success flag and the current frame converted to BGR
#             return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         else:
#             return (ret, None)
#     else:
#         return (ret, None)

# def snapshot():
#     ret, frame = get_frame()
#     if ret:
#         img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         drawDetectionBoxes(idxs, boxes, classIDs, confidences, img)
#         cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", img)

def changeOnHover(button, colorOnHover, colorOnLeave): 
    button.bind("<Enter>", func=lambda e: button.config( 
        background=colorOnHover)) 
  
    # background color on leving widget 
    button.bind("<Leave>", func=lambda e: button.config( 
        background=colorOnLeave))

start=False

def strtv():
    global start
    start=True
    mainwork()

def endv():
    global start 
    start = False
#global vehicle_count

vehicle_count=0 

def mainwork():

    FRAMES_BEFORE_CURRENT = 15  

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    #Using GPU if flag is passed
    if USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()] for yolov4
    x1_line = 0
    y1_line = video_height//2
    x2_line = video_width
    y2_line = video_height//2
    previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
    # previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
    num_frames = 0
    global vehicle_count
    start_time = int(time.time())

    # btn_snapshot=Button(root, text="Snapshot",bg='yellow',width=45,pady=15,command=snapshot)
    # btn_snapshot.grid(row=8,column=1)

    root.configure(bg='LightBlue1')

    startVideo = Button(root, text="Play",bg='green2',pady=15,command=strtv)
    startVideo.grid(row=9,column=0,sticky="nsew")
    pauseVideo= Button(root, text="Pause",bg='gold',pady=15,command=endv)
    stopVideo= Button(root, text="Stop",bg='red',pady=15,command=root.destroy)
    pauseVideo.grid(row=9,column=1,sticky="nsew")
    stopVideo.grid(row=9,column=2,sticky="nsew")

    wi,he = video_width,video_height

    if video_width > 1700 or video_height > 1700:
        wi=video_width//3
        he=video_height//3

    # if video_height > 1700 :
    #     he=video_height//3

    elif video_width > 1000 or video_height > 1000:
        wi=video_width//2
        he=video_height//2

    # if video_height > 1000 :
    #     he=video_height//2

    startVideo.config(borderwidth=5,relief=RAISED)
    pauseVideo.config(borderwidth=5,relief=RAISED)
    stopVideo.config(borderwidth=5,relief=RAISED)

    changeOnHover(startVideo, "PaleGreen2", "green2") 
    changeOnHover(pauseVideo, "goldenrod1", "gold") 
    changeOnHover(stopVideo, "tomato", "red")

    global start
    global mask
    global polygon_coords
    
    while start:
        num_frames+= 1
        #print("FRAME:\t", num_frames)
        # Initialization for each iteration
        boxes, confidences, classIDs = [], [], [] 

        #Calculating fps each second
        start_time, num_frames = displayFPS(start_time, num_frames)
        # read the next frame from the file
        (grabbed, frame) = cap.read()
        
        #Drawing ROI
        jump_pnts = np.array(polygon_coords, np.int32)
        cv2.polylines(frame, [jump_pnts], True,(0,0,255),3)
        
        #Creating the mask
        masked = cv2.bitwise_and(frame,mask)        

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(masked, 1 / 255.0, (inputWidth, inputHeight),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for i, detection in enumerate(output):
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > preDefinedConfidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    #Printing the info of the detection
                    #print('\nName:\t', LABELS[classID],
                    #   '\t|\tBOX:\t', x,y)

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,preDefinedThreshold)


        vehicle_count, current_detections, count = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)

        displayVehicleCount(frame, vehicle_count, count)

        myLabel0 = Label(root,text='Vans: ' + str(count['van']),borderwidth=3,relief=RAISED)
        myLabel0.grid(row=2,column=0,pady=10,sticky="nsew")
        myLabel0['bg']="CadetBlue1"
        myLabel1 = Label(root,text='Bicycles: ' + str(count['bicycle']),borderwidth=3,relief=RAISED)
        myLabel1.grid(row=0,column=0,pady=10,sticky="nsew")
        myLabel1['bg']="CadetBlue1"
        myLabel2 = Label(root,text='Motorbikes: '+ str(count['motorbike'] ),borderwidth=3,relief=RAISED)
        myLabel2.grid(row=0,column=1,pady=10,sticky="nsew")
        myLabel2['bg']="SkyBlue1"
        myLabel3 = Label(root,text='Autorickshaws: ' + str(count['auto']),borderwidth=3,relief=RAISED)
        myLabel3.grid(row=0,column=2,pady=10,sticky="nsew")
        myLabel3['bg']="CadetBlue1"
        myLabel4 = Label(root,text='Cars: ' + str(count['car']),borderwidth=3,relief=RAISED)
        myLabel4.grid(row=1,column=0,pady=10,sticky="nsew")
        myLabel4['bg']="CadetBlue1"
        myLabel5 = Label(root,text='Trucks: '+ str(count['truck']),borderwidth=3,relief=RAISED)
        myLabel5.grid(row=1,column=1,pady=10,sticky="nsew")
        myLabel5['bg']="SkyBlue1"
        myLabel6 = Label(root,text='Rickshaws: ' + str(count['rickshaw']),borderwidth=3,relief=RAISED)
        myLabel6.grid(row=1,column=2,pady=10,sticky="nsew")
        myLabel6['bg']="CadetBlue1"
        myLabel7 = Label(root,text='Minitrucks: ' + str(count['minitruck']),borderwidth=3,relief=RAISED)
        myLabel7.grid(row=2,column=1,pady=10,sticky="nsew")
        myLabel7['bg']="SkyBlue1"
        myLabel8 = Label(root,text='Total vehicles: ' + str(vehicle_count),fg='red',borderwidth=10,relief=RAISED)
        myLabel8.grid(row=3,column=1,pady=10,sticky="nsew")
        myLabel8['background']="LightPink1"
        myLabel9 = Label(root,text='Buses: ' + str(count['bus']),borderwidth=3,relief=RAISED)
        myLabel9.grid(row=2,column=2,pady=10,sticky="nsew")
        myLabel9['bg']="CadetBlue1"
        
        app=Frame(root)
        app.grid(row=4,columnspan=4,sticky="nsew")
        lmain = Label(app,pady=10)
        lmain.grid(row=4,column=0,rowspan=4,columnspan=4,sticky="nsew")

        row_count = 0
        
        while row_count !=3:
            Grid.rowconfigure(root,row_count,weight=1)
            row_count+=1

        col_count = 0
        
        while col_count !=3:
            Grid.columnconfigure(root,col_count,weight=1)
            col_count+=1

        _, frame = cap.read()

        #Drawing ROI
        cv2.polylines(frame, [jump_pnts], True,(0,0,255),3)

        (wa,ha,ma) = frame.shape

        # frame = cv2.resize(frame,(ha//2,wa//2),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        # print(frame.shape)


        #Creaing the mask
        #masked = cv2.bitwise_and(frame,mask)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        drawDetectionBoxes(idxs, boxes, classIDs, confidences, cv2image)

        img = Image.fromarray(cv2image)

        h1,w1=root.winfo_height(),root.winfo_width()
        # print(root.bbox(0,4))

        # (aa,bb,cc,dd) =root.bbox(0,4)

        img = img.resize((wi*w1//800,he*h1//1000))#size of video

        # img = img.resize((cc*3,bb))
        # print(img.size)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

        # Updating with the current frame detections
        previous_frame_detections.pop(0) #Removing the first frame from the list
        # previous_frame_detections.append(spatial.KDTree(current_detections))
        previous_frame_detections.append(current_detections)


root = Tk()
#global vehicle_count
root.grid()
root["bg"] = "gainsboro"
root.title('Count Vehicle')
root.minsize(400,500) 

# variable = StringVar(root)
# variable.set("Select video to play :") # default value

# w = OptionMenu(root, variable, "rourkela.mp4", "walking.avi", "video3", "video4")
# w.grid(row=5,column=1)

######### for recorded video
def myvid():
    root.filename = filedialog.askopenfilename(initialdir="",title="Select video:",filetypes=(("mp4 files", "*.mp4"),("avi files", "*.avi"),("MPG files", "*.MPG")))
    global selection
    if len(root.filename):
        selection.set(str(root.filename))
    global file_name
    file_name = Label(root,textvariable=selection).grid(row=2,column=1)
    if selection.get().endswith(('.mp4','.avi','.MPG')):
        global startVideo
        startVideo = Button(root, text="Start",bg='cyan3',width=10,pady=10,command=my_show)
        startVideo.grid(row=6,column=1)       
        startVideo.config(borderwidth=3,relief=RAISED)
        changeOnHover(startVideo, "cyan4", "cyan3") 
        global openvideo
        openvideo = True
    else:
        # #global startVideo
        # startVideo = Button(root, text="Start",bg='cyan3',width=30,pady=15,command=my_cam)
        # startVideo.grid(row=3,column=1)       
        # startVideo.config(borderwidth=5,relief=RAISED)
        # changeOnHover(startVideo, "cyan4", "cyan3") 
        # #global openvideo
        # openvideo = True
        if not openvideo:
            selection.set('No video selected, please select a video !')

    # print(root.winfo_height(),root.winfo_width())

def my_show():
    # global selection
    # selection = str(variable.get())
    global cap
    cap = cv2.VideoCapture(selection.get())
    global video_width
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    global video_height
    video_height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video_width,video_height)

    selection.set("")
    startVideo.destroy()
    selectVideo.destroy()
    
    #Declaring Globals
    global mask
    global polygon_coords
    #Getting a temp frame
    pd = PolygonDrawer("Polygon")
    grabbed, temp_frame = cap.read()
    pd.run(temp_frame)
    
    polygon_coords = pd.points
    polygon_coords.pop()
    #Creating the mask 
    mask = MakeMask(polygon_coords,temp_frame.shape)
    strtv()

openvideo = False
selection = StringVar()



# for USB CAM
def myusb():
   
    global startVideo
    startVideo = Button(root, text="Start",bg='cyan3',width=10,pady=10,command=my_cam)
    startVideo.grid(row=6,column=1)       
    startVideo.config(borderwidth=3,relief=RAISED)
    changeOnHover(startVideo, "cyan4", "cyan3") 
    global openvideo
    openvideo = True

def my_cam():
    # global selection
    # selection = str(variable.get())
    global cap
    cap = cv2.VideoCapture(0)
    global video_width
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    global video_height
    video_height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video_width,video_height)

    selection.set("")
    startVideo.destroy()
    selectVideo.destroy()
    
    #Declaring Globals
    global mask
    global polygon_coords
    #Getting a temp frame
    pd = PolygonDrawer("Polygon")
    grabbed, temp_frame = cap.read()
    pd.run(temp_frame)
    
    polygon_coords = pd.points
    polygon_coords.pop()
    #Creating the mask 
    mask = MakeMask(polygon_coords,temp_frame.shape)
    strtv()

openvideo = False
selection = StringVar()


# for USB CAM
def myIP():
   
    global startVideo
    startVideo = Button(root, text="Start",bg='cyan3',width=10,pady=10,command=my_IP)
    startVideo.grid(row=6,column=1)       
    startVideo.config(borderwidth=3,relief=RAISED)
    changeOnHover(startVideo, "cyan4", "cyan3") 
    global openvideo
    openvideo = True

def my_IP():
    # global selection
    # selection = str(variable.get())
    global cap
    cam1 = input('Enter the user name: ')
    cam2 = input('Enter the password: ')
    cam3 = input('Enter the url of the camera: ')   #Enter integer 0 in case of Webcam
    cam='rtsp://'+cam1+':'+cam2+'@'+cam3+':554/PSIA/streaming/channels/102'
    print(cam)
    cap = cv2.VideoCapture(cam)
    global video_width
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    global video_height
    video_height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video_width,video_height)

    selection.set("")
    startVideo.destroy()
    selectVideo.destroy()
    
    #Declaring Globals
    global mask
    global polygon_coords
    #Getting a temp frame
    pd = PolygonDrawer("Polygon")
    grabbed, temp_frame = cap.read()
    pd.run(temp_frame)
    
    polygon_coords = pd.points
    polygon_coords.pop()
    #Creating the mask 
    mask = MakeMask(polygon_coords,temp_frame.shape)
    strtv()

openvideo = False
selection = StringVar()


# print(root.geometry())
# fe = Label(root,text="---------------------")
# fe.grid(row=1,column=0,sticky="nsew")
# fe["bg"]="gainsboro"


# FOR recorded video
selectVideo = Button(root, text="Select video",bg='SkyBlue1',width=10,pady=10,command=myvid)
selectVideo.grid(row=3,column=1,sticky="nsew")
selectVideo.config(borderwidth=2,relief=RAISED)
changeOnHover(selectVideo,'white','SkyBlue1')

# FOR USB WEBCAM
selectVideo = Button(root, text="Select camera",bg='NavajoWhite2',width=10,pady=10,command=myusb)
selectVideo.grid(row=4,column=1,sticky="nsew")
selectVideo.config(borderwidth=2,relief=RAISED)
changeOnHover(selectVideo,'white','NavajoWhite2')

# FOR IP CAMERA
selectVideo = Button(root, text="Select IPcamera",bg='PaleGreen2',width=10,pady=10,command=myIP)
selectVideo.grid(row=5,column=1,sticky="nsew")
selectVideo.config(borderwidth=2,relief=RAISED)
changeOnHover(selectVideo,'white','PaleGreen2')
# fhe = Label(root,text="---------------------")
# fhe.grid(row=1,column=2,sticky="nsew")
# fhe["bg"]="gainsboro"

# print(root.winfo_height(),root.winfo_width())


row_count = 0
        
while row_count !=3:
    Grid.columnconfigure(root,row_count,weight=1)
    row_count+=1

root.mainloop()