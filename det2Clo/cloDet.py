import traceback
import pprint
pp = pprint.pprint

from jSona import jSona

from detectron2.data             import MetadataCatalog
from detectron2.engine           import DefaultPredictor

class cloDet() :
    def __init__(self, cfg, MODEL_PATH, LABEL_PATH, threshold=0.8, batchsize=512) :
        self.cfg    = cfg
        self.jso    = jSona()
        self.labels = self.jso.loadJson(LABEL_PATH)
        self.classes= {v:k for k,v in self.labels.items()}

        self.cfg.MODEL.WEIGHTS                        = MODEL_PATH
        self.cfg.DATASETS.TEST                        = ('detect',)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = threshold
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batchsize
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES          = len(self.labels)

        MetadataCatalog.get('detect').set(thing_classes=list(self.labels.keys()))

        self.predictor = DefaultPredictor(cfg)

    def detect(self, img, show=True, scale=0.5, bias_width=500, save_path=None) :
        # tw = bias_width; th = int(img.shape[0]*tw/img.shape[1])
        # if tw > 100 : img = cv2.resize(img, dsize=(tw, th), interpolation=cv2.INTER_LINEAR)
        outputs = self.predictor(img)
        return outputs

    def meta(self) :
        return MetadataCatalog.get('detect')

    def error(self, e, msg="", isExit=False) :
        print("ERROR {} : {}".format(msg, e))
        traceback.print_exc()
        if isExit : exit()  