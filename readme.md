# yolo3-tensorflow 
TensorFlow implementation of yolo v3 objects detection.  
Based: full or tiny, and cnn or mobilenets(mobilenet_v1, mobilenet_v2);  
We can get 6 combination, but 1 of them has a little parameters and performs badly. 
So, you should build these 5 combination as folloing:
* cnn + full
* cnn + tiny
* mobilenet_v1 + full
* mobilenet_v2 + full
* mobilenet_v2 + tiny 

These 5 frameworks are provided in this repository.

## Dependence
 python3  
 tensorflow == 1.13  
 opencv3  

## Quick start
* cnn full yolo3
  1. Download official [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put it on `model_data` floder of project.
  2. Run the command `python convert_weights.py full`  to convert weights to TensorFlow checkpoint file, which will locate in `logs/cnn_full/` and named `cnn_full_model.data-00000-of-00001`
  3. Run the command `python yolo.py` or `python yolo.py -w logs/cnn_full/cnn_full_model`  and input the image path to detect.
  4. Detect example:  
     <img src="images/full.jpg"/>
* cnn tiny yolo3
  1. Download official [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights) and put it on `model_data` floder of project.
  2. Run the command `python convert_weights.py tiny`  to convert weights to TensorFlow checkpoint file, which will locate in `logs/cnn_tiny/` and named `cnn_tiny_model.data-00000-of-00001`
  3. Run the command `python yolo.py -w logs/cnn_tiny/cnn_tiny_model` and input the image path to detect.
  4. Detect example:  
     <img src="images/tiny.jpg"/>

## Train

1. Prepare Dataset  
Before training, you should generate your own annotation file and class names file.
One row for one image  
Row format: image_file_path box1 box2 ... boxN  
Box format: x_min,y_min,x_max,y_max,class_id (no space)  
For VOC dataset, try `python util/voc_annotation.py`  
For your own dataset, you should change the [voc_annotation.py](voc_annotation.py)  
Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Prepare yolo anchors   
run `python util/kmeans.py` to generate anchors. Note that, anchor number 
should be 9 if you wang to train full yolo, else it should be 6. 

3. Start to train  
The train arguments can be seen in [config/train_config.py](config/train_config.py).      
    ```
    usage: train.py [-h] [-n NET_TYPE] [-t TINY] [-b BATCH_SIZE] [-e EPOCH]
                    [-lr LEARN_RATE] [-pt PRETRAIN_PATH]
                    [--anchor_path ANCHOR_PATH] [--train_path TRAIN_PATH]
                    [--classes_path CLASSES_PATH] [-d DEBUG]
    
    optional arguments:
      -h, --help            show this help message and exit
      -n NET_TYPE, --net_type NET_TYPE
                            net type: cnn, mobilenetv1 mobilenetv2 or mobilenetv3
      -t TINY, --tiny TINY  whether tiny yolo or not
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            batch_size
      -e EPOCH, --epoch EPOCH
                            epoch
      -lr LEARN_RATE, --learn_rate LEARN_RATE
                            learn_rate
      -pt PRETRAIN_PATH, --pretrain_path PRETRAIN_PATH
                            pretrain path
      --anchor_path ANCHOR_PATH
                            anchor path
      --train_path TRAIN_PATH
                            train file path
      --classes_path CLASSES_PATH
                            classes path
      -d DEBUG, --debug DEBUG
                            whether print per item loss
    ```
    The dafault framework is cnn + full. If you want to train others, you can pass 
    the `-n` (cnn, mobilenetv1 or mobilenetv2) and `-t` (True or False) arguments.

4. To be simple  
   I have write scripts in `shell` folder. Just run `CUDA_VISIBLE_DEVICES='0' sh ./shell/train_cnn_full.sh` or
   `CUDA_VISIBLE_DEVICES='0' nohup stdbuf -oL sh ./shell/train_cnn_full.sh > logs/cnn_full.txt &` in background and the log 
   will be write in cnn_full.txt.  
   You can also change some other arguments such as batch_size and epoch and so on.  
   If you want to use pretrain, you should pass the pretrain path. I will provide the pretrain weights later.
5. NOTE
    The mobilenet is converged more slower than cnn, you should train more epoch.
6. Tensorboard
    You can use Tensorboard to watch the training trend.  
    Run `Tensorboard --logdir ./  --host 127.0.0.1`   
    you can see mAP score
     <img src="images/mAP.png"/>
7.  test your training weights with your test datasets  
    `python test.py`  
    you maybe need to change configs in config/pred_conf.py
    
    
## Predict
The prediction arguments can be seen in [config/pred_config.py](config/pred_config.py).
```
usage: yolo.py [-h] [-i IMAGE] [-v VIDEO] [-w WEIGHT_PATH] [--score SCORE]
               [--classes_path CLASSES_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        image path
  -v VIDEO, --video VIDEO
                        video path
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        weight path
  --score SCORE         score threshold
  --classes_path CLASSES_PATH
                        classes path
 ```
 Note that, the weights filename should be like `cnn_full_model.xxx`, `cnn_tiny_model.xxx`, or others. 
 the framework will be built by the word 'cnn' and 'full' or 'cnn' and 'tiny'.  
 You can predict an image or video.  
 For example:  
 `python yolo.py -w weight_path`  
 `python yolo.py -i imgage_path -w weight_path`  
 `python yolo.py -v video_path -w weight_path`