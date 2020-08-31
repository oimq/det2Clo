# System libraries
import numpy as np
import random
import threading as th
import traceback
import pandas as pd
from queue   import Queue
from os.path import join
from os      import listdir, makedirs
from pprint  import pprint as pp 
from cv2     import cv2
from tqdm    import tqdm

# User libraries
from .clo2Utils import typc, error
from jSona import save, load, dumps

# Detectron2 libraries
from detectron2.data             import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures       import BoxMode
from detectron2.config           import get_cfg
from detectron2.engine           import DefaultTrainer
from detectron2                  import model_zoo

# Logging modules
import logging
logger = logging.getLogger("Trainer")
logger.setLevel(logging.INFO)
logger_formatter      = logging.Formatter("- %(levelname)s %(asctime)s | %(name)s | %(message)s")
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setFormatter(logger_formatter)
logger.addHandler(logger_stream_handler)
def log(msg :any ="", level :str ="i") : 
    global logger
    if   type(msg) == type([]) : msg = ", ".join(msg) 
    elif type(msg) == type(()) : msg = " : ".join(msg)
    if   level == "i" : logger.info( "\033[34m{}\033[0m".format(msg))
    elif level == "e" : logger.error("\033[91m{}\033[0m".format(msg))

# System functions


# Datasets manipulating and parsing the result
class Factory() :
    def __init__(self :type) :
        pass

    def check_missing_data(self, anno_list :list, imgs_list :list, check_ratio :float =0.2, cry :bool =True, prefix :str ="") :
        sample_anno = list(map(lambda x:x+prefix, random.sample(anno_list, int(len(anno_list)*check_ratio))))

        log("Check the missing data. total size : {}, sample size : {}, ratio : {:.3f}%.".format(
                len(anno_list), len(sample_anno), check_ratio*100), 'i')
        if cry : pbar = tqdm(total=len(sample_anno))

        check_count = 0
        for sample in sample_anno : 
            check_count += 1 if sample in imgs_list else 0

            if cry : pbar.update(1)
        if cry : pbar.close()
        if check_count != len(sample_anno) :
            log("{:.3f}% images are missing.".format((1-(check_count/len(sample_anno)))*100), 'i')
            return False
        else :
            log("There is no missing data.")
            return True

    # anno_list : annotation names, anno_data : annotation data!
    def th_get_deep_info(self, anno_list :list, anno_data :list, imgs_path :str, index_range :tuple, pbar=None) :
        for ainx in range(*index_range) : 
            anno_data[ainx]['image_id']  = int(anno_list[ainx].split('.')[0])
            anno_data[ainx]['image_src'] = join(imgs_path, anno_list[ainx].split('.')[0]+'.jpg')
            anno_data[ainx]['shape']     = list(cv2.imread(anno_data[ainx]['image_src']).shape)
            if pbar : pbar.update(1)

    def get_deep(self, TRAN_PATH, sample_size :int =0, check_ratio=0.02, cry :bool =True) :
        anno_path, imgs_path = join(TRAN_PATH, "annos/"), join(TRAN_PATH, "image/")
        anno_list, imgs_list = sorted(listdir(anno_path)), sorted(listdir(imgs_path))
        if sample_size > 0 : anno_list = anno_list[:sample_size]
        
        if self.check_missing_data(anno_list, imgs_list, check_ratio) :
            log("Load deepfashion datasets - please wait a few seconds...", 'i')
            anno_data = [load(join(anno_path, anno_list[i])) for i in range(len(anno_list))]
            log("Load deepfashion datasets - Done : {} numbers.".format(len(anno_data)), 'i')

            log("Supply the additional information.", 'i')
            pbar = tqdm(total=len(anno_data)) if cry else None

            # Using threading!
            NUM_THREAD  = 4
            chunk_size  = [len([0 for i in range(len(anno_data))][i::NUM_THREAD]) for i in range(NUM_THREAD)]
            thread_list = [
                th.Thread(
                    target=self.th_get_deep_info, 
                    args=(anno_list, anno_data, imgs_path, (sum(chunk_size[:i]), sum(chunk_size[:i+1]), 1), pbar)) for i in range(NUM_THREAD)
            ]
            for thread in thread_list : thread.start()
            for thread in thread_list : thread.join()

            if cry : pbar.close()
            return anno_data
        else :
            return None

    # Create datasets for deepfasion
    def datasets4deep(self, TRAN_PATH :str, categories, sample_size :int =0, check_ratio=0.02, cry=True) :
        log("Getting deepfashion data")
        deep_data, datasets = self.get_deep(TRAN_PATH, sample_size, check_ratio, cry), list()

        log("Create the datasets from deepfashion.")
        if cry : pbar = tqdm(total=len(deep_data))
        for deep in deep_data :
            item_keys = set(deep.keys())^set(['source', 'pair_id', 'image_src', 'image_id', 'shape'])
            annotations = list()
            for ikey in item_keys :
                annotations.append({
                    'bbox'          :   deep[ikey]['bounding_box'],
                    'bbox_mode'     :   BoxMode.XYXY_ABS,
                    'segmentation'  :   deep[ikey]['segmentation'],
                    'category_id'   :   categories['{} {:02}'.format(deep[ikey]['category_name'], deep[ikey]['style'])],
                    'iscrowd'       :   0,
                })
            datasets.append({
                'file_name'     :   deep['image_src'],
                'image_id'      :   deep['image_id'],
                'height'        :   deep['shape'][0],
                'width'         :   deep['shape'][1],
                'annotations'   :   annotations
            })
            if cry : pbar.update(1)
        if cry : pbar.close()
        return datasets

    def categories_deep(self, TRAN_PATH :str, sample_size :int =0, cry :bool =True) :
        anno_path = join(TRAN_PATH, "annos/")
        anno_list = sorted(listdir(anno_path))
        if sample_size > 0 : anno_list = anno_list[:sample_size]

        log("Load deepfashion datasets - please wait a few seconds...", 'i')
        anno_data = [load(join(anno_path, anno_list[i])) for i in range(len(anno_list))]
        log("Load deepfashion datasets - Done : {} numbers".format(len(anno_data)), 'i')

        log("Get categories from datasets", 'i')
        if cry : 
            pdar = tqdm(bar_format="{desc}", desc="Initializing...", position=0)
            pbar = tqdm(total=len(anno_data), position=1)
        categories = dict()
        for anno in anno_data :
            item_keys = set(anno.keys())^set(['source', 'pair_id'])
            for ikey in item_keys :
                lkey = anno[ikey]['category_id']*100+anno[ikey]['style']
                if lkey not in categories : categories[lkey] = '{} {:02}'.format(anno[ikey]['category_name'], anno[ikey]['style'])
            if cry : 
                pdar.set_description(desc="Getting label : {}, {}".format(lkey, categories[lkey]))
                pbar.update(1)
        if cry : 
            pbar.close(); pdar.close()
        
        if categories : categories = {v:i for i,v in enumerate(sorted(list(categories.values())))}
        return categories

    # @ TRAN_PATH : getting for train images
    # @ ANNO_PATH : getting for train descriptions
    # @ DICT_PATH : original id -> category name
    def get_imat(self, TRAN_PATH :str, ANNO_PATH :str, LABL_PATH :str, 
                 sample_size :int =0, check_ratio :float =0.02, cry :bool =True, classes :list =[]) :

        log("Load iMaterialist datasets - please wait a few seconds...")
        anno_data = pd.read_csv(ANNO_PATH)
        log("Load iMaterialist datasets - Done : {} numbers".format(len(anno_data)))

        log("Extract ids and Convert category ids to dictionary for train-sets")
        if classes : anno_data = anno_data[anno_data['ClassId'].isin(classes)]
        anno_data = list(anno_data.to_dict('index').values())
        if sample_size > 0 : anno_data = anno_data[:sample_size]
        anno_list, imgs_list = sorted([anno['ImageId'] for anno in anno_data]), sorted(listdir(TRAN_PATH))
        cate_dict = self.dictionary_imat(LABL_PATH, classes)
        log("Extraction and Conversion about original class {} are done.".format(classes))

        if self.check_missing_data(anno_list, imgs_list, check_ratio, cry, prefix='.jpg') :
            

            # Using threading!
            NUM_THREAD  = 8
            chunk_data  = [list() for i in range(NUM_THREAD)]
            log("{} Threads are prepared for get informations".format(NUM_THREAD))
            for i in range(len(anno_data)) : chunk_data[i%NUM_THREAD].append(anno_data[i])
            pbar = tqdm(total=len(anno_data)) if cry else None
            anno_data = dict()
            que = Queue()
            thread_list = [
                th.Thread(
                    target=lambda q, arg1, arg2, arg3, arg4 : q.put(self.th_get_imat_info(arg1, arg2, arg3, arg4)), 
                    args=(que, TRAN_PATH, chunk_data[i], cate_dict, pbar)
                ) for i in range(NUM_THREAD)
            ]

            for thread in thread_list : thread.start()
            for thread in thread_list : thread.join()
            if cry : pbar.close()

            log("Success to make infomation data. Now we integrate the results")
            while not que.empty() :
                results = que.get()
                for result in results :
                    if result['image_id'] in anno_data : anno_data[result['image_id']].append(result)
                    else : anno_data[result['image_id']] = [result]
            log("Integration is done. The total number of information is {}.".format(len(anno_data)))
            return anno_data
        else :
            return None

    # Create the iMaterist Datasets
    def datasets4imat(self, TRAN_PATH :str, ANNO_PATH :str, LABL_PATH :str, categories, sample_size :int =0, check_ratio :float =0.2, cry :bool =True, classes :list =[]) :
        log("Getting iMaterialist data")
        imat_data, datasets = self.get_imat(TRAN_PATH, ANNO_PATH, LABL_PATH, sample_size, check_ratio, cry, classes), list()

        log("Create the datasets from iMaterialist.")
        if cry : pbar = tqdm(total=len(imat_data))
        for imat_items in imat_data.values() : # imat : [{~~}. {~~}]
            annotations = list()
            for imat in imat_items :
                annotations.append({
                    'bbox'          :   imat['bbox'],
                    'bbox_mode'     :   BoxMode.XYXY_ABS,
                    'segmentation'  :   imat['segmentation'],
                    'category_id'   :   categories[imat['category_name']],
                    'iscrowd'       :   0,
                })
            datasets.append({
                'file_name'     :   imat['image_src'],
                'image_id'      :   imat['image_id'],
                'height'        :   imat['height'],
                'width'         :   imat['width'],
                'annotations'   :   annotations
            })
            if cry : pbar.update(1)
        if cry : pbar.close()
        return datasets

    # @ Main reason is that data is constructed by 'id'.
    # @ So, we should have route like 'id' -> 'categories name' -> 'start from 0- id'
    def dictionary_imat(self, LABL_PATH :str, classes :list =[]) :
        labels = load(LABL_PATH)['categories']
        if classes : return {label['id']:label['name'] for label in labels if label['id'] in classes}
        else       : return {label['id']:label['name'] for label in labels}

    def categories_imat(self, LABL_PATH :str, classes :list =[]) :
        labels = load(LABL_PATH)['categories']
        if classes : return {v['name']:i for i,v in enumerate(list(filter(lambda x:x['id'] in classes, labels)))}
        else       : return {v['name']:i for i,v in enumerate(labels)}

    # @ The iMat needs bbox, xyxy coordinates from encoded pixels.
    # @ So, the function has abandoned features and support threading.
    # @  1) basics     : image_id, file_name, width, height.
    # @  2) converts   : bbox, xyxy
    # @  3) integrates : integrate all results to a dictionary.
    def th_get_imat_info(self, TRAN_PATH :str, anno_data :list, cate_dict :dict, pbar =None) :
        imat_info = list()
        for ainx in range(len(anno_data)) :
            segs, bbox = self.convertPixels(
                anno_data[ainx]['EncodedPixels'], 
                anno_data[ainx]['Height'], 
                anno_data[ainx]['Width']
            )
            info = {
                'image_id'  : anno_data[ainx]['ImageId'],
                'image_src' : join(TRAN_PATH, anno_data[ainx]['ImageId']+".jpg"),
                'width'     : anno_data[ainx]['Width'],
                'height'    : anno_data[ainx]['Height'],
                'bbox'      : bbox,
                'segmentation' : segs,
                'category_id'  : anno_data[ainx]['ClassId'],
                'category_name': cate_dict[anno_data[ainx]['ClassId']]
            }
            imat_info.append(info)
            if pbar : pbar.update(1)
        return imat_info

    def rle2BBOX(self, epixels, height, width):
        pairs = np.fromiter(epixels.split(), dtype=np.uint).reshape((-1, 2)) # pairs = a
        X, Y0, Y1 = pairs[:,0]//height, (pairs[:,0]%height), (pairs[:,0]%height)+pairs[:,1]
        return min(X), min(Y0), max(X), max(Y1)

    # rle example : 6068157 7 6073371 20 6078584 34 -> convert -> (x0, y0, x1, y1)
    def rle2XYXY(self, epixels, height, width):
        mask = np.full(shape=(height*width, 1), fill_value=0, dtype=np.uint8)
        annotation = [int(x) for x in epixels.split(' ')]
        for i, sp in enumerate(annotation[::2]) : mask[sp:sp+annotation[2*i+1]] = 80
        mask = mask.reshape((height, width), order='F')
        return (mask).astype(np.uint8)

    def convertPixels(self, epixels, height, width) :
        mask = self.rle2XYXY(epixels, height, width)#; return mask
        bbox = self.rle2BBOX(epixels, height, width)#; return bbox
        bbox = np.array(bbox, dtype=np.uint8).tolist()
        segs = []
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = np.array(contour.flatten().tolist(), dtype=np.uint8).tolist()
            if len(contour) > len('XYXY') : segs.append(contour)
        return (segs, bbox)

# Managing the datasets, models and metrics. Also show images
class Manager() :
    def __init__(self :type, DATA_PATH :str, YAML_NAME :str, CONF_PATH :str) :
        self.configuration(DATA_PATH, YAML_NAME, CONF_PATH)
        self.categories = None
        self.classes    = None
        self.datasets   = None

    def check_train_materials(self) :
        if self.categories and self.datasets : 
            log("We have {} numbers categories and {} numbers datasets.".format(len(self.categories), len(self.datasets)), 'i')
            return True
        if not self.categories : log("There is no categories.", 'e')
        if not self.datasets   : log("There is no datasets.", 'e')
        return False

    def load_categories(self, CATE_PATH :str) :
        self.categories = load(CATE_PATH)
        self.classes    = {v:k for k,v in self.categories.items()}

    def load_datasets(self, SETS_PATH :str) :
        self.datasets = load(SETS_PATH)

    def configuration(self, DATA_PATH, YAML_NAME, CONF_PATH) :
        config = load(CONF_PATH)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(YAML_NAME))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(YAML_NAME)
        self.cfg.DATASETS.TRAIN                          = ("train",)
        self.cfg.DATASETS.TEST                           = ()
        self.cfg.DATALOADER.NUM_WORKERS                  = config['DATALOADER']['NW'] if 'DATALOADER' in config else 2
        self.cfg.SOLVER.IMS_PER_BATCH                    = config['SOLVER']['BATCH']  if 'SOLVER'     in config else 2
        self.cfg.SOLVER.BASE_LR                          = config['SOLVER']['LR']     if 'SOLVER'     in config else 0.00025
        self.cfg.SOLVER.MAX_ITER                         = config['SOLVER']['ITER']   if 'SOLVER'     in config else 30000
        self.cfg.OUTPUT_DIR                              = config['OUTPUT_DIR']       if 'OUTPUT_DIR' in config else './output/'
        return self.cfg

# Training the datasets
class Trainer() :
    @classmethod
    def __init__(self :type, mgr :Manager) :
        self.mgr = mgr

    def training(self, BATCH_SIZE_PER_IMAGE :int, cry=True) :
        try :
            if self.mgr.cfg == None : error(Exception("We don't have configuration for training. sorry."), "TRAINING", ex=True)
            if not self.mgr.check_train_materials() : error(Exception("So, we cannot progress the training. sorry.", "TRAINING", ex=True))
            log("Load categories and trains for training.")
            categories = self.mgr.categories
            datasets   = self.mgr.datasets
            log("Loaded {} numbers categories and {} numbers trains.".format(len(categories), len(datasets)))

            self.mgr.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE    = BATCH_SIZE_PER_IMAGE
            self.mgr.cfg.MODEL.ROI_HEADS.NUM_CLASSES             = len(categories)

            log("Register the trainsets and set the matedata.") ########
            DatasetCatalog.clear()
            DatasetCatalog.register("train", lambda d=datasets:d)
            MetadataCatalog.get('train').set(thing_classes=list(categories.keys()))
            log("Registering is Done.")

            log("Training would be started after a few minutes...") ########
            makedirs(self.mgr.cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(self.mgr.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            return True

        except Exception as e:
            error(e, "ERROR TRAINING", cry=True, ex=True)
            return False