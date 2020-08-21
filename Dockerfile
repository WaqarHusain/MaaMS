FROM python:3
# Add all the files related to service.py here.
WORKDIR /home/ncai/Complex-YOLOv3/


ADD ./service.py /
ADD ./service_utils.py /app/service_utils.py
ADD ./checkpoints/yolov3_ckpt_epoch-298.pth  /app/yolov3_ckpt_epoch-298.pth

ADD ./data/classes.names /app/classes.names

ADD ./utils/utils.py  /
ADD ./utils/kitti_bev_utils.py /app/kitti_bev_utils.py
ADD ./utils/kitti_aug_utils.py /app/kitti_aug_utils.py
ADD ./models.py /app/models.py

# Mention work directory
WORKDIR /home/ncai/Complex-YOLOv3

COPY ./ ./

# Update the requirements.txt.

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./app

ENTRYPOINT [ "python" ]

CMD [ "service.py" ]