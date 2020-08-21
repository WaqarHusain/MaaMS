import numpy as np
import utils.kitti_utils as kitti_utils
import utils.kitti_aug_utils as aug_utils
import utils.kitti_bev_utils as bev_utils
import os

package_dir = os.pardir

LOCALHOST = '127.0.0.1'
PORT = 8080
CONNECTION_LIMIT = 1
SIZE_LIMIT = 1073741824 # Represents 2^30, 2 to the power 30.

weights_path = os.path.join(package_dir,'checkpoints/yolov3_ckpt_epoch-298.pth')
class_path = os.path.join(package_dir ,'data/classes.names')
model_cfg = os.path.join(package_dir ,'config/complex_yolov3.cfg')

CONF_THRESHOLD = 0.5 #default
NMS_THRESHOLD  = 0.5 #default

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    predictions = np.zeros([50, 7], dtype=np.float32)
    count = 0
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            predictions[count, :] = cls_pred, x/img_size, y/img_size, w/img_size, l/img_size, im, re
            count += 1

    predictions = bev_utils.inverse_yolo_target(predictions, cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = aug_utils.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):

        str = "Pedestrian"
        if l[0] == 0:str="Car"
        elif l[0] == 1:str="Pedestrian"
        elif l[0] == 2: str="Cyclist"
        else:str = "DontCare"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h,obj.w,obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))
    
        _, corners_3d = kitti_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_utils.read_labels_for_bevbox(objects_new)    
        if not noObjectLabels:
            labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P) # convert rect cam to velo cord

        target = bev_utils.build_yolo_target(labels)
        utils.draw_box_in_bev(RGB_Map, target)

    return objects_new