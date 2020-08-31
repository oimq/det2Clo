# System libraries
import traceback
from cv2 import cv2
from pprint import pprint as pp

# User libraries
from jSona        import save, load
from .clo2Utils   import typc, error
from .clo2Trainer import Manager

# Detectron2 libraries
from detectron2.data    import MetadataCatalog
from detectron2.engine  import DefaultPredictor

# Logging modules
import logging
logger = logging.getLogger("Detector")
logger.setLevel(logging.INFO)
logger_formatter      = logging.Formatter("- %(levelname)s line %(lineno)d | %(asctime)s | %(name)s | \n%(message)s")
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setFormatter(logger_formatter)
logger.addHandler(logger_stream_handler)
def log(msg :any ="", level :str ="i") : 
    global logger
    if   type(msg) == type([]) : msg = ", ".join(msg) 
    elif type(msg) == type(()) : msg = " : ".join(msg)
    if   level == "i" : logger.info( "\033[34m{}\033[0m".format(msg))
    elif level == "e" : logger.error("\033[91m{}\033[0m".format(msg))

# Define Detector
class Detector() :
    def __init__(self :type, mgr :Manager, MODL_PATH, CATE_PATH, threshold=0.8, batchsize=512) :
        self.categories = load(CATE_PATH)
        self.classes    = {v:k for k,v in self.categories.items()}
        self.predictor  = None
        self.meta       = None
        self.makePredictor(mgr.cfg, MODL_PATH, threshold, batchsize)

    def makePredictor(self :type, cfg, MODL_PATH :str, threshold :float, batchsize :int) :
        try :
            cfg.MODEL.WEIGHTS                        = MODL_PATH
            cfg.DATASETS.TEST                        = ('detect',)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = threshold
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batchsize
            cfg.MODEL.ROI_HEADS.NUM_CLASSES          = len(self.categories)
            MetadataCatalog.get('detect').set(thing_classes=list(self.categories.keys()))
            self.predictor = DefaultPredictor(cfg)
            self.meta      = MetadataCatalog.get('detect')

        except Exception as e :
            error(e, "MAKEPREDICTOR")
            self.predictor = None

    def detect(self, img, resize_width :int =0, spec :bool = True) :
        if   typc(img, None) : error(Exception("There is no image data."), "DETECT"); return None
        elif typc(img, "")   : img = cv2.imread(img)
        if resize_width > 0  : 
            img = cv2.resize(
                img, 
                dsize=(resize_width, int(img.shape[0]*resize_width/img.shape[1])), 
                interpolation=cv2.INTER_LINEAR
            )
            
        if typc(self.predictor, None) : error(Exception("We don't have the predictor."), "DETECT"); return None
        if spec : 
            outputs = {
                'outputs'  : self.predictor(img),
                'metadata' : self.meta if self.meta else None
            }
        else :
            outputs = self.predictor(img)
        return outputs

    def class2label(self, cn) :
        if not typc(cn, 0) : error(ValueError('Wrong Class number type.', "CLASS2LABEL")); return None
        if self.classes : return self.classes[cn]
        else            : return None

    def parse4specs(self :type, segments) :
        if not typc(segments, dict()) : error(ValueError('Wrong Segments type.', "PARSE4SEGMENTS")); return None
        instances = segments['outputs']['instances']
        parsed = {
            'boxes'  : [box.tolist() for box in list(instances.get('pred_boxes').to('cpu'))],
            'scores' : [sco.tolist() for sco in list(instances.get('scores').to('cpu'))],
            'classes': [cla.tolist() for cla in list(instances.get('pred_classes').to('cpu'))],
            'masks'  : [mas.tolist() for mas in list(instances.get('pred_masks').to('cpu'))],
            'shape'  : instances.image_size,
        }
        parsed.update({
            'class' : [self.class2label(cn) for cn in parsed['classes']]
        })
        return parsed