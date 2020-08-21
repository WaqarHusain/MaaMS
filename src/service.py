import numpy as np
import math
import os , sys

#Setting project path to access utils package
sys.path.append(os.pardir)

import json
import time
import torch
from utils import config as cnf
import utils.utils as utils
from models import *
from service_utils import *
from utils import kitti_utils as kitti_utils
from utils import kitti_aug_utils as aug_utils
from utils import kitti_bev_utils as bev_utils
import socket
from utils import config as cnf
from queue import Queue
from struct import unpack

def model_inference(model , frame):
    '''
    Takes point cloud as n-dim numpy array, and forward pass through Complex YOLO Model.
    '''       
    no_points = int(frame.shape[0]/4)
    frame = frame.reshape(no_points,4)

    # Reducing number of points 
    b = bev_utils.removePoints(frame, cnf.boundary)
    #Converting pointcloud into BEV Representation.
    bev_maps = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
    #Convert to
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                        
    img = torch.from_numpy(bev_maps).type(torch.FloatTensor).to(device)
    
    #Insert batch size 1.
    img = img.view(1,img.size(0),img.size(1),img.size(2))
    
    #Eval mode
    model.eval()

    # Get detections 
    with torch.no_grad():
        detections = model(img)
        detections = utils.non_max_suppression_rotated_bbox(detections, CONF_THRESHOLD, NMS_THRESHOLD) 
    
    img_detections = []  # Stores detections for each image index
    img_detections.extend(detections)

    RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
    RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
    RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
    RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
    
    RGB_Map *= 255
    RGB_Map = RGB_Map.astype(np.uint8)
    dict_detection = {}
        
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        try:
            detections = utils.rescale_boxes(detections, cnf.BEV_WIDTH, RGB_Map.shape[:2])
        except:
            print('')
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            
            dict_detection[float(cls_pred)] = { 'x': float(x) , 'y': float(y) , 'w': float(w) , 'l':float(l), 
                'im': float(im) ,'conf':float(conf)
            }
    
    #Converting python dictionary into json.
    json_dict = json.dumps(dict_detection)
    #Convert it into binary format.
    results   = json.loads(json_dict.decode('utf-8'))

    return results            

if __name__ == '__main__':

    print('Initializing Model............')
    #Initializing Model
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = utils.load_classes(class_path)
    # Set up model
    model = Darknet(model_cfg, cnf.BEV_HEIGHT).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))
    print('Starting Server...............')
    #Creating a server on localhost IPV4 connection using TCP stream.
    with  socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serv:
        #bind server to localhost at specified port
        serv.bind((LOCALHOST, PORT))
         #Currently listening to a single connection
        serv.listen(CONNECTION_LIMIT)
        flag = False
        while True:           
            conn, addr = serv.accept()
            # While connected. 
            with conn:
                while True:
                    start_time = time.time()
                    #First receive the package length.
                    package_length = conn.recv(8)
                    data = b''
                    (length,) = unpack('>Q',package_length)
                    # Loop over if the frame is partially received.
                    while len(data) < length:
                        # If the data is recevied in batches.
                        to_read = length - len(data)
                        data += conn.recv(4096 if to_read > 4096 else to_read)
                        
                    if not data: 
                        break
                    try:
                        frame = np.frombuffer(data,dtype=np.float32)
                        flag = True
                    except:
                        output = 'Uncomplete Buffer Size:'
                        flag = False

                    if flag == True:
                        
                        output =  model_inference(model,frame)
                        
                    message = str.encode(output)
                    # Pass
                    conn.send(message)
                    end_time = time.time()
                    print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
 