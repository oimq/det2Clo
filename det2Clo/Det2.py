import os
import numpy as np
import random
import threading as th
import traceback
import pprint
pp = pprint.pprint
from cv2    import cv2
from tqdm   import tqdm

from .visIm import visIm
from .cloDet import cloDet
from jSona import jSona

from detectron2.data             import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures       import BoxMode
from detectron2.config           import get_cfg
from detectron2.engine           import DefaultTrainer
from detectron2                  import model_zoo

class Det2() :
    def __init__(self) :
        self.vim = visIm()
        self.jso = jSona()
        self.cld = None
        self.cfg = None
        self.cm  = ColorMode.IMAGE

    def class2LABEL(self, class_num) :
        if self.cld : return self.cld.classes[class_num]
        else        : return 'none'

    def parse_segments(self, segments) :
        if not segments : return None
        instances = segments['outputs']['instances']
        parsed = {
            'boxes'  : [box.tolist() for box in list(instances.get('pred_boxes').to('cpu'))],
            'scores' : [sco.tolist() for sco in list(instances.get('scores').to('cpu'))],
            'classes': [cla.tolist() for cla in list(instances.get('pred_classes').to('cpu'))],
            'masks'  : [mas.tolist() for mas in list(instances.get('pred_masks').to('cpu'))],
            'shape'  : instances.image_size,
        }
        parsed.update({
            'labels' : [self.class2LABEL(class_num) for class_num in parsed['classes']]
        })
        return parsed
    
    def detect(self, img, scale=1.0, color_mode=ColorMode.IMAGE, meta=None, parse=False) : 
        if self.cld == None : raise Exception("There is no clothing detector. Please make detector first.")
        if meta == None : meta = self.cld.meta()
        self.img = cv2.imread(img) if type(img) == type("") else img
        if type(img) == type(None)  : return None
        detects = {
            'outputs'   : self.cld.detect(self.img),
            'scale'     : scale,
            'cm'        : color_mode,
            'metadata'  : meta,
        }
        return detects

    def configuration(self, DATA_PATH, YAML_NAME, CONFIGS) :
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(YAML_NAME))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(YAML_NAME)
        self.cfg.DATASETS.TRAIN                          = ("train",)
        self.cfg.DATASETS.TEST                           = ()
        self.cfg.DATALOADER.NUM_WORKERS                  = CONFIGS['DATALOADER']['NW']      if 'DATALOADER' in CONFIGS else 2
        self.cfg.SOLVER.IMS_PER_BATCH                    = CONFIGS['SOLVER']['IPB']         if 'SOLVER'     in CONFIGS else 2
        self.cfg.SOLVER.BASE_LR                          = CONFIGS['SOLVER']['BL']          if 'SOLVER'     in CONFIGS else 0.00025
        self.cfg.SOLVER.MAX_ITER                         = CONFIGS['SOLVER']['MI']          if 'SOLVER'     in CONFIGS else 30000
        self.cfg.OUTPUT_DIR                              = CONFIGS['OUTPUT_DIR']            if 'OUTPUT_DIR' in CONFIGS else 10

        return self.cfg

    def makeDetector(self, MODEL_PATH, LABEL_PATH, threshold=0.3, batchsize=512) :
        if self.cfg == None : raise Exception("There is no configurations. Please do the configuration first.")
        self.cld = cloDet(
            cfg=self.cfg, 
            MODEL_PATH=MODEL_PATH, 
            LABEL_PATH=LABEL_PATH,
            threshold=threshold,
            batchsize=batchsize
        )

    def training(self, DATA_PATH, BATCH_SIZE_PER_IMAGE, cry=True) :
        try :
            if self.cfg == None : raise Exception("We don't have configuration for trains. sorry.")

            if cry : print("# Load labels and trains for training.") ########
            labels = self.jso.loadJson(DATA_PATH+'train_descriptions.json')
            trains = self.jso.loadJson(DATA_PATH+'train_deepfashions.json')
            if cry : print("# Done.") ########

            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE    = BATCH_SIZE_PER_IMAGE
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES             = len(labels)

            if cry : print("# Register the trainsets and set the matedata.") ########
            DatasetCatalog.clear()
            DatasetCatalog.register("train", lambda d=trains:d)
            MetadataCatalog.get('train').set(thing_classes=list(labels.keys()))
            if cry : print("# Done.") ########

            if cry : print("# Training would be started after a few minutes...") ########
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(self.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            return True
        except Exception as e:
            print("ERROR TRAINING. :", e)
            traceback.print_exc()
            return False

    def labels_deep(self, data_path, sample_size=None, cry=True) :
        anno_path = os.path.join(data_path, "annos/")
        anno_list = sorted(os.listdir(anno_path))
        if sample_size : anno_list = anno_list[:sample_size]

        if cry : print("# Load deepfashion datasets - please wait a few seconds...") ########
        anno_data = [self.jso.loadJson(os.path.join(anno_path, anno_list[i])) for i in range(len(anno_list))]
        if cry : print("# Done.") ########

        if cry : print("# Get labels from datasets"); pbar = tqdm(total=len(anno_data)) ########
        labels = dict()
        for anno in anno_data :
            item_keys = set(anno.keys())^set(['source', 'pair_id'])
            for ikey in item_keys :
                lkey = anno[ikey]['category_id']*100+anno[ikey]['style']
                if lkey not in labels : labels[lkey] = '{} {:02}'.format(anno[ikey]['category_name'], anno[ikey]['style'])
            if cry : pbar.update(1) ########
        if cry : pbar.close() ########
        return list(labels.values())
            
    def get_deep(self, data_path, sample_size=None, check_ratio=0.02, cry=True) :
        anno_path, imgs_path = os.path.join(data_path, "annos/"), os.path.join(data_path, "image/")
        anno_list, imgs_list = sorted(os.listdir(anno_path)), sorted(os.listdir(imgs_path))
        if sample_size : anno_list = anno_list[:sample_size]
        
        if self.check_missing_data(anno_list, imgs_list, ratio=check_ratio) :
            if cry : print("# Load deepfashion datasets - please wait a few seconds...") ########
            anno_data = [self.jso.loadJson(os.path.join(anno_path, anno_list[i])) for i in range(len(anno_list))]
            if cry : print("# Done.") ########
            if cry : print("# Supply the additional information.") ########
            pbar = tqdm(total=len(anno_data)) if cry else None ########
            # Using threading!
            NUM_THREAD  = 8
            chunk_size  = [len([0 for i in range(len(anno_data))][i::NUM_THREAD]) for i in range(NUM_THREAD)]
            thread_list = [
                th.Thread(
                    target=self.th_get_images_info, 
                    args=(anno_list, anno_data, imgs_path, (sum(chunk_size[:i]), sum(chunk_size[:i+1]), 1), pbar)) for i in range(NUM_THREAD)
            ]
            for thread in thread_list : thread.start()
            for thread in thread_list : thread.join()

            if cry : pbar.close(); print() ########
            return anno_data
        else :
            return None

    def datasets_iMat(self) :
        pass

    def datasets_deep(self, data_path, labels, sample_size=None, check_ratio=0.02, cry=True) :
        deep_data = self.get_deep(data_path, sample_size, check_ratio, cry)
        datasets  = list()
        if cry : print("# Create the datasets from deepfashion."); pbar = tqdm(total=len(deep_data)) ########
        for deep in deep_data :
            item_keys = set(deep.keys())^set(['source', 'pair_id', 'image_src', 'image_id', 'shape'])
            annotations = list()
            for ikey in item_keys :
                annotations.append({
                    'bbox'          :   deep[ikey]['bounding_box'],
                    'bbox_mode'     :   BoxMode.XYXY_ABS,
                    'segmentation'  :   deep[ikey]['segmentation'],
                    'category_id'   :   labels['{} {:02}'.format(deep[ikey]['category_name'], deep[ikey]['style'])],
                    'iscrowd'       :   0,
                })
            datasets.append({
                'file_name'     :   deep['image_src'],
                'image_id'      :   deep['image_id'],
                'height'        :   deep['shape'][0],
                'width'         :   deep['shape'][1],
                'annotations'   :   annotations
            })
            if cry : pbar.update(1) ########
        if cry : pbar.close(); print()########
        return datasets

    def check_missing_data(self, anno_list, imgs_list, ratio=0.02, cry=True) :
        sample_anno = list(map(lambda x:x.split('.')[0], random.sample(anno_list, int(len(anno_list)*ratio))))
        criteo_imgs = "/".join(imgs_list)
        check_count = 0
        if cry :  ########
            print("# Check the missing data. total size : {}, sample size : {}, ratio : {:.3f}%.".format(
                len(anno_list), len(sample_anno), ratio*100))
            pbar = tqdm(total=len(sample_anno))
        for sample in sample_anno : 
            if cry : pbar.update(1) ########
            check_count += 1 if sample in criteo_imgs else 0
        if cry : pbar.close(); print()########
        if check_count != len(sample_anno) :
            print("Some data are missing. Missing ratio : {:.4f}.\n".format(1-(check_count/len(sample_anno))))
            return False
        return True

    # anno_list : annotation names, anno_data : annotation data!
    def th_get_images_info(self, anno_list, anno_data, imgs_path, index_range, pbar=None) :
        for ainx in range(*index_range) : 
            anno_data[ainx]['image_id']     = int(anno_list[ainx].split('.')[0])
            anno_data[ainx]['image_src']    = os.path.join(imgs_path, anno_list[ainx].split('.')[0]+'.jpg')
            anno_data[ainx]['shape']        = list(cv2.imread(anno_data[ainx]['image_src']).shape)
            if pbar : pbar.update(1)


    # bbox should be [ x0, y0, x1, y1 ]
    def showBBOX(self, img, bbox, color=255) :
        img = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
        img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] = color
        self.vim.show(img, show_type='plt')

    def showCONT(self, img, contours) :
        img = img + np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
        img = cv2.drawContours(img, contours, -1, (255,255,255), 10)
        self.vim.show(img, show_type='plt')

    # segment should be [ x0, y0, ... , xn, yn ]
    def showSEGS(self, img, segments) :
        cpimg = img.copy()
        for segment in segments : cpimg[self.seg4MASK(segment, cpimg.shape) >= 255] = 0
        self.vim.show(cpimg, show_type='plt')
    
    def seg4MASK(self, segment, shape, color=(255,255,255), thickness=5) :
        mask = np.full(shape=shape, fill_value=0, dtype=np.uint8)
        poly = segment.reshape((-1, 2)).reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img=mask, pts=[poly], isClosed=True, color=color, thickness=thickness)
        return (mask).astype(np.uint8)

    def img2MASK(self, img, masks) :
        masked_imgs = list()
        if type(masks) == type([]) : masks = np.array(masks)
        for mask in masks :
            cpimg = img.copy()
            cpimg[mask==False] = 255
            masked_imgs.append(cpimg)
        return masked_imgs

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
        segs = []
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            print(contour)
            contour = contour.flatten().tolist()
            if len(contour) > len('XYXY') : segs.append(contour)
        return [segs, bbox]

    def compareMetrics(self, METC_PATH, fields_x=['iteration'], fields_y=['total_loss'], cry=True) :
        metrics = self.jso.loadJson(os.path.join(METC_PATH, 'metrics.json'))
        metrics = list(filter(lambda metric:metric['iteration']!=0 and (metric['iteration']+1)%5000==0, metrics))

        indice = {v:i for i,v in enumerate(metrics[0].keys())}
        matrix  = np.array([list(metric.values()) for metric in metrics])
 
        fxs = [list(matrix[:, indice[field_x]]) for field_x in fields_x] 
        fys = [list(matrix[:, indice[field_y]]) for field_y in fields_y]
        
        if cry : 
            if  len(fxs) == 1  : 
                self.vim.plot(fxs[0], fys[0], fields_x[0], fields_y[0])

            else    : 
                self.vim.plots(fxs, fys, fields_x, fields_y)
        mm = { fields_y[i]: (fys[i].index(max(fys[i])), max(fys[i]), 
                             fys[i].index(min(fys[i])), min(fys[i])) for i in range(len(fys)) }
        return mm
