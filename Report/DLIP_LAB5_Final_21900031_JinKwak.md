# LAB: Unmanned Ground Sensor System

**Date:**  2024.06.24

**Author:**  Jin Kwak/ 21900031, Yoonseok Choi/22100747

**Course:** Image Processing with Deep Learning

**About:** Unmanned Ground Sensor System

**[Github](https://github.com/Kwak-Jin/DLIP)**

**[Demo Video](https://youtu.be/UDJZypf0S9I)**

---

# Introduction
## 1. Objective
This lab is to replace manpower for alert duty in military using mono-camera.

For human&camera replacement, the system should be able to detect the depth and class of an object(North Korean) using a mono camera.

As in Figure 1, the image of a single-eyed(Mono) camera does not have depth information and this disables a user to derive distance of object from the camera. This problem is one of the major problem in vision industry that should be overcome.

By using deep learning model, the distance of an object from a camera is derived like in figure 2.

<p align ='center'> <img src="../Report_image/Final Report/CamCoord.png" alt="FinalLab.drawio" style="zoom:150%;" /> Figure 1. Camera Coordinate </p>

<p align ='center'> <img src="../Report_image/Final Report/스크린샷 2024-06-18 023334.png" alt="FinalLab.drawio" style="zoom:120%;" /> Figure 2. Distance Visualization  </p>


### Goal: Depth/Object detection in the military region 

### Reason For the Topic
- Due to low birth rate, reduction in military power.
- Guard Duty Burden
- Reduced Combat power due to non-combat(Alert Duty) missions(Severe Weather)
- Guard duty automation (may be used for other purposes such as prison)

### Expected results
1.  **Military personnel resources optimization**
2. Can work as double-checking tool and black box

## 2. Preparation

### Development Environment
- **Python** 3.10
- **CUDA**: 12.1
- **IDE**: Jetbrains Pycharm
- **Deep Learning Model**
  1. Mono-Depth Estimation: [Metric3D v2](https://arxiv.org/pdf/2404.15506v1)
  2. Object Detection: [YOLO V8](https://www.ultralytics.com/yolo)

### Software Installation
#### 1. Miniconda Install

For the compact and easy use, [Miniconda Installation](https://docs.anaconda.com/free/miniconda/miniconda-install/) is installed for the program. 

#### 2. Python Install
In the project, [Click here to install Python in local](https://www.python.org/downloads/windows/). Recommended version is python >= 3.9

#### 3. Create Virtual Environment
```shell
conda create -n UGSS python=3.10
conda activate UGSS
```
#### 4. Packages Download

To use CUDA, one should identify their GPU specification. [Check GPU spec](https://en.wikipedia.org/wiki/CUDA) for correct Pytorch installation and [CUDA installation](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local). 

``` shell
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Dataset  

- The program is designed for real-time application. The North Korean is replaced with [e.g. teddy bear](https://www.google.com/search?sca_esv=07dd5598d35cffb5&sca_upv=1&q=teddy+bear&udm=2&fbs=AEQNm0DmKhoYsBCHazhZSCWuALW8l8eUs1i3TeMYPF4tXSfZ9zKNKSjpwusJM2dYWg4btGKvTs8msUkFt41RLL2EsYFXj1HJ-6Tz3zY-OaA8p5OIwLlYAhqYgKeQsybVCfK3TClp5eJ8pKyvjHPuKkxzOkfs39PPooyb18QionBChgkg3bORCI0L1Q6BO3S5b3bJfdHG6epm&sa=X&sqi=2&ved=2ahUKEwilvsT5juGGAxU7qVYBHQKNAzQQtKgLegQIChAB&biw=1080&bih=1785&dpr=1). Click the link for teddy bear image.


# Algorithm
## 1. Overview

The flow of the program is described in figure 3. 

<p align ='center'> <img src="../Report_image/Final Report/FinalLab.drawio.png" alt="FinalLab.drawio" style="zoom:150%;" /> Figure 3. Flowchart of the program</p>




## 2. Procedure

### Preprocessing
1. Set intrinsic matrix for the camera. 
2. Convert image to 1064 x 616 size
3. Padding the outline of the image to be the same size as the actual image.
4. Normalize by subtracting the mean from the padding image and dividing by the standard deviation.

###  Classification

There are 5 objects that should be detected.

1. Person (person that is not North Korean)
2. Bicycle
3. Car
4. Motorcycle
5. North Korean (In YOLO pretrained model, there is no label "North Korean", therefore North Korean is replaced to Teddy Bear)

As in the flowchart(Figure 3.), the object prediction is required to detect object. For each classes, the decision is differently made.
This is done by using YOLO v8 s model.

### Decision Making

There are two options:
1. Execution of enemy
   - When enemy is detected and close enough to aim
2. Siren and report
   - When enemy is detected but too far to aim
   - If any vehicles are in the security region(Aware)

    
# Result and Discussion
## Classification Evaluation
For detection evaluation, Confusion Matrix for multiple classes is drawn in Figure 4.
Where classes:       
A: Person   
B: Bicycle  
C: Cars     
D: Motorcycle   
E: Teddy-bear   

<p align ='center'> <img src="../Report_image/Final Report/FinalConfusionMatrix.png" alt="FinalLab.drawio" style="zoom:150%;" /> Figure 4. Confusion Matrix of Classes</p>

Classes Human and Teddy bear are detected in the case very well whereas in real life, the one should be able to classify them with very strict precision.
In figure 4, accuracy for both motorcycle and bicycle is low. However, the decision-making for both of the classes is the same therefore, it does not really affect the system.

The metrics (precision, recall, F1 score, F0.5Score and F2 score) are calculated for each classes in Figure 5.
<p align ='center'> <img src="../Report_image/Final Report/PrecisionRecallF1F2F0_5Scores.png" alt="FinalLab.drawio" style="zoom:170%;" /> Figure 5. Classifications Metrics for each classes</p>

## Range Evaluation
For accurate distance measure, Laser Distance Meter Tester is used(Bosch DLE 70).
<p align='center'><img src = "https://img.danawa.com/prod_img/500000/910/921/img/921910_1.jpg?shrink=330:*&_v=20210507135605"/> Figure 6. Meter Tester </p>

<p align='center'> <img src="../Report_image/Final Report/KakaoTalk_20240624_001617944_01.jpg" alt="FinalLab.drawio" style="zoom:120%;" /> Figure 7. Distance Testing </p>



​	The range evaluation is done by calculating the average distance from the model(Metric3D) and the true distance measured by Meter Tester. The result in Figure 8 shows how accurate the model is. The error increases when distance increases.

​	In fact, the teddy bear detection range is about 4.0 m. Therefore, the maximum range that can be applied is 4 m and at minimum of 0.5m.

<p align='center'> <img src="../Report_image/Final Report/RangeValid.png" alt="FinalLab.drawio" style="zoom:150%;" /> Figure 8. Table for Range Validation </p>

## 1. Final Result
**[Demo Video Embedded](https://youtu.be/UDJZypf0S9I)**

We have programmed automatic ground sensor system that can replace human powered security.

This program is a real-time applicable program that includes deep learning object detection and depth estimation, decision-making, and control. 

## 2. Discussion
1. As the military is well-protected region, classification is very significant in the program. To overcome, use higher model in YOLO. As YOLO x model is at minimum 100 Hz real-time system, the object accuracy is more important than real-time, it is okay to use higher(heavier) model.
<p align="center"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"> Figure 9. mean Average Precision </p>



# Conclusion
## Lab Purpose Achievement
<p align='center'> <img src="../Report_image/Final Report/FAR.png" alt="FinalLab.drawio" style="zoom:120%;" /> Figure 10. Enemy far </p>

<p align='center'> <img src="../Report_image/Final Report/CLOSE.png" alt="FinalLab.drawio" style="zoom:120%;" /> Figure 11. Enemy close  </p>

​	Object detection and distance estimation is well applied in the program with the range of 0.5m to 4.0m. Average loop time consumed is  50Hz for both model prediction and decision making and execution on i7-13700k, GeForce 4070Ti. Therefore, this satisfies real-time application.

Therefore, the algorithm satisfies the objectives of the lab.


## Improvement
1. It is necessary to check whether it is possible to detect enemies in camouflaged situations outdoors.
2. To avoid accidents such as misdetection or no detection of North Korean, the threshold should be conservatively adjusted.
3. During the actions(Gun sound and Siren respectively), as this is a sequential process, other lines of code is stopped. To make it act like a real-time system, The actions should be done in other process via sending flags by using shared memory or UDP communication.


---
# Appendix
**Main Code**
```python
dependencies = ['torch', 'torchvision']

import os
import torch
from ultralytics import YOLO
# import cv2 as cv
import pygame
import time
from enum import Enum
import numpy as np

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model

metric3d_dir = os.path.dirname(__file__)

MODEL_TYPE = {
    'ConvNeXt-Tiny':  {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
    },
    'ConvNeXt-Large': {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
    },
    'ViT-Small':      {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
    },
    'ViT-Large':      {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
    },
    'ViT-giant2':     {
        'cfg_file':  f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
    },
}


def metric3d_convnext_tiny(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_convnext_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_vit_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d_vit_giant2(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']
    
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
                torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
                strict=False,
        )
    return model


def metric3d(model: torch.nn.Module, rgb_origin: np.ndarray) -> np.ndarray:
    intrinsic = [1126.6, 1130.7, 977.7, 571.0]
    
    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT,
                             value=padding
                             )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    
    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    
    ###################### canonical camera space ######################
    # inference
    model.cuda().eval()
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
    
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                 pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear'
                                                 ).squeeze()
    ###################### canonical camera space ######################
    
    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)
    
    #### normal are also available
    # if 'prediction_normal' in output_dict:  # only available for Metric3Dv2, i.e. vit model
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    #     normal_confidence = output_dict['prediction_normal'][:, 3, :, :]  # see https://arxiv.org/abs/2109.09881 for details
    #     # un pad and resize to some size if needed
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1],
                  pad_info[2]: pred_normal.shape[2] - pad_info[3]]
    #     # you can now do anything with the normal
    #     # such as visualize pred_normal
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    cv2.imshow('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))
    
    return pred_depth.cpu().numpy()


play_list = (("mixkit-arcade-chiptune-explosion-1691.wav"),
             ("mixkit-police-siren-1641.wav"))


class Status(Enum):
    NOK = 0
    YOK = 1
    FAR = 2
    CLOSE = 3


CLASS_NORTH_KOREAN = 77
CLASS_PERSON = 0
CLASS_BICYCLE = 1
CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
CLASSES = [CLASS_PERSON, CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE, CLASS_NORTH_KOREAN]

if __name__ == '__main__':
    import cv2
    import numpy as np
    
    DISTANCE_THRESHOLD = 2.0
    pygame.init()
    model_detect = YOLO('yolov8s.pt')
    
    explosion_sound = pygame.mixer.Sound(play_list[0])
    siren_sound = pygame.mixer.Sound(play_list[1])
    
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    status_flag = Status.NOK
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    
    while ret:
        depth = metric3d(model, frame)
        frame_detect = frame
        result_detect = model_detect.predict(frame, classes=CLASSES, half=True)
        
        cv2.imshow("Pure Frame", result_detect[0].plot())
        class_cpu = result_detect[0].boxes.cls.detach().cpu().numpy()
        bbox_coords_cpu = result_detect[0].boxes.xyxy.to('cpu').numpy()
        
        for bbox_idx in range(len(bbox_coords_cpu)):
            bbox = bbox_coords_cpu[bbox_idx]
            # If there is a vehicle in the image, the control department should give a warning(Siren)
            if class_cpu[bbox_idx] in CLASSES[1:3]:
                status_flag = Status.FAR
            elif class_cpu[bbox_idx] == CLASS_NORTH_KOREAN:
                print('North Korean invaded!')
                middle_point = (int((bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2))
                depth_middle = depth[middle_point]
                if (depth_middle > DISTANCE_THRESHOLD):
                    status_flag = Status.FAR
                else:
                    status_flag = Status.CLOSE
                print("Depth of the middle point", depth_middle)
                break
        
        if status_flag == Status.FAR:
            siren_sound.play()
            time.sleep(3)
            pygame.mixer.music.set_volume(1.0)
        elif status_flag == Status.CLOSE:
            explosion_sound.play()
            time.sleep(1)
            pygame.mixer.music.set_volume(1.0)
        
        status_flag = Status.NOK
        
        if cv2.waitKey(10) == 27:
            break
        ret, frame = cap.read()
    
    pygame.quit()
    cv2.destroyAllWindows()
    cap.release()
```



