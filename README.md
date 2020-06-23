## det2Clo

##### Image categorization and segmentation based on detectron2

for installing and using, requirements are below : 

* numpy : https://github.com/numpy/numpy

* opencv-python : https://github.com/skvark/opencv-python

* tqdm : https://github.com/tqdm/tqdm

* pytorch : https://github.com/pytorch/pytorch

* pillow : https://github.com/python-pillow/Pillow

* detectron2 : https://github.com/facebookresearch/detectron2

* jSona : https://github.com/oimq/jSona

***

### Installation

The pre-install resource is jSona.

It is recommended to run in a <b>cuda based gpu environment.</b>

I used deepfashion2 datasets for the clothings detector. Thanks for the favor.
https://github.com/switchablenorms/DeepFashion2


```code
pip3 install det2Clo-master/
```

***

### Projects

Before we started, notice that the det2Clo separates in three parts.

* cloDet
    * detect : Detect categories and masking segments

* visIm
    * show : Show the image, it is okay to add detects.
    
* Det2
    * convertPixels : Get [ x0, y0, x1, y1 ] coordinates from run-length encoded segment regions.
    * training : Training a detector.
    * labels_deep : Make the labels(annotation) data for deepfashion2
    * get_deep : Create annotation data <image_id, image_src, shape> for deepfashion2
    * datasets_deep : Create the training datasets for deepfashion2 
        
    * class2LABEL : Convert class number to label
    * detect : Detect object from cloDet.
    * parse_segments : Parsing the detectron2 segmenting results to more readable.
    * showBBOX : Show the bbox and image from result of detection.
    * showCONT : Show the contours and image from result of detection.
    * showSEGS : Show the segments and image from result of detection.
    * seg4MASK : Get object masks from result of detection.
    * img2MASK : Get images applied masking of objects. 
    * compareMetrics : Get train-compared(result) graphs like loss, accuracy, ... etc.
    
  




***

### Examples

* Script for training
```python3
from det2Clo import det2
from jSona import jSona

det2 = Det2() 
jso = jSona()

DATA_PATH = 'deepfashion2/'
TRAN_PATH = DATA_PATH+'train/'

# We should have the annotations data
train_labels   = det2.get_deep(get_deep)
train_datasets = det2.datasets_deep(TRAN_PATH, train_labels, check_ratio=0.1)
jso.saveJson(DATA_PATH+'train_deepfashions.json', train_datasets)    

YAML_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'

CONFIGS = {
    'DATALOADER':{
        'NW':4,
    },
    'SOLVER':{
        'IPB':4, 'BL':0.0002, 'MI':150000,
    },
    'OUTPUT_DIR':TRAN_PATH+'output/'
}

det2.configuration(DATA_PATH, YAML_NAME, CONFIGS)
det2.training(DATA_PATH, 512)

# Show the training results by graphs.
METC_PATH = TRAN_PATH+'metrics/'
det2.compareMetrics(METC_PATH)
```

* Outputs
```python3

```

* Script for detections
```python3
from det2Clo import det2

det2 = Det2() 

MODEL_PATH = 'model/'
LABEL_PATH = 'label/'
IMAGE_PATH = 'sample.jpg'

det2.makeDetector(MODEL_PATH, LABEL_PATH)
detect_results = det2.detect(IMAGE_PATH)
det2.showSEGS(IMAGE_PATH, detect_results)
```

* Outputs
```python3

```


***


### Notices

###### Unauthorized distribution and commercial use are strictly prohibited without the permission of the original author and the related module.
