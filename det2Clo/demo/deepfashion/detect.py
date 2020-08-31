from os.path import join
from pprint  import pprint as pp
from det2Clo import clo2Imagine  as c2i
from det2Clo import clo2Trainer  as c2t
from det2Clo import clo2Detector as c2d

if __name__=="__main__" :
    # ! Define the paths
    DATA_PATH = '/home/denny/datasets/deepfashion/'
    CONF_PATH = join(DATA_PATH, 'conf/train_config.json')
    YAML_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    MODL_PATH = '/home/denny/myImages/Teshionista/outputs/deep/model_0054999.pth'
    CATE_PATH = join(DATA_PATH, 'train_descriptions.json')

    # ! Make manager for get configuration
    manager = c2t.Manager(DATA_PATH, YAML_NAME, CONF_PATH)

    # ! Make detector for detect clothings
    detector = c2d.Detector(manager, MODL_PATH, CATE_PATH, threshold=0.3, batchsize=512)
    results = detector.detect(join(DATA_PATH, 'sample.jpg'), resize_width=-1, spec=True)
    pp(results['outputs'])

    # ! Make stage for show the detected image
    stage = c2i.Stage()
    stage.show(join(DATA_PATH, 'sample.jpg'), tool="cv2", scale=0.5, detects=results)