# MaaMS (Model as a Micro Service)
### The module will contain the following features: 
##### 1 - Object Detection in Virtual LiDAR Data Stream. 
##### 2 - Deployment of Trained DL Model for Object Detection in LiDAR Data as a Micro-Service (MaaMS) 
##### 3 - Implementation of Publish/Subscribe Messaging Protocol 4 - Dummy implementation of Virtual LiDAR using KITTI Dataset

##### To Run the code
     1. $ cd src/
     2. $ python service.py
### Note
##### You have to first download the Yolo weights, which can be done by running checkpoints/downloads_weight.py .

##### Task Pending.
1. Conversion of coordinates from point cloud to unity image format.
2. Update service.py to check for weights, if not available first download it using checkpoints/downloads_weight.py
