# System libraries
from cv2 import cv2
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

# User libraries
from .clo2Utils import typc
from jSona import load

# Detectron2 libraries
from detectron2.utils.visualizer import Visualizer, ColorMode

# Logging modules
import logging
logger = logging.getLogger("Imagine")
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

class Stage() :
    TOOLS = ['plt', 'cv2']
    
    def __init__(self) :
        pass
        
    def draw(self, img, tool :str ='plt', shape :tuple =(15, 15), detects :list =None) :
        if   tool == self.TOOLS[0] :
            if img.shape[2] == 3 : img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=shape)
            plt.axis('off')
            plt.imshow(img)
            plt.show()

        elif tool == self.TOOLS[1] :
            cv2.imshow(winname='The Visualizing Window', mat=img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show(self, img =None, tool :str ="plt", scale :float =0.5, detects=None) :
        if tool not in self.TOOLS : return print("There are no type for [ {} ]".format(tool))

        if   typc(img, "")   : img = cv2.imread(img)
        elif typc(img, None) : log("There are no image.", 'e')
        if detects :
            if len(set(detects.keys())&set(['outputs', 'metadata'])) == 2 :
                visual = Visualizer(img[:, :, ::-1], metadata = detects['metadata'], scale = scale)
                prediction = visual.draw_instance_predictions(detects['outputs']['instances'].to('cpu'))
                img = prediction.get_image()[:, :, ::-1]
            else : log("detects must have {} and be specified... but your is {}".format(['outputs', 'metadata'], detects.keys()), 'e')
        self.draw(img, tool)

    def paint(self, img, detects, scale :float =0.5, ) :
        if   typc(img, "")   : img = cv2.imread(img)
        elif typc(img, None) : log("There are no image.", 'e')
        if len(set(detects.keys())&set(['outputs', 'metadata'])) == 2 :
            visual = Visualizer(img[:, :, ::-1], metadata = detects['metadata'], scale = scale)
            prediction = visual.draw_instance_predictions(detects['outputs']['instances'].to('cpu'))
            return prediction.get_image()[:, :, ::-1]
        else : log("detects must have {} and be specified... but your is {}".format(['outputs', 'metadata'], detects.keys()), 'e')
        
    def plot(self, xx, yy, title :str ="", xlabel :str ="", ylabel :str ="", shape :tuple =(15, 15), guide =True) :
        # Adjust figure
        plt.figure(figsize=shape)
        if title : plt.title(title)
        if xlabel : plt.xlabel(xlabel, fontsize=shape[0])
        if ylabel : plt.ylabel(ylabel, fontsize=shape[1])
        # Plot the points
        plt.xticks(xx, xx)
        plt.yticks(yy, ["{:.3f}".format(y) for y in yy])
        main_line_style  = {'linestyle':'-',  'color':'k', 'linewidth':5}
        plt.plot(xx, yy, **main_line_style)
        # Appear the guide lines
        if guide :
            guide_line_style = {'linestyle':'--', 'color':'c', 'linewidth':1}
            for xinx in range(len(xx)) : plt.plot([xx[xinx]]*2, [min(yy), yy[xinx]], **guide_line_style)
            for yinx in range(len(yy)) : plt.plot([min(xx), xx[yinx]], [yy[yinx]]*2, **guide_line_style)
        # Show!
        plt.show()

    def plots(self, xxx, yyy, title :str ="", xlabels :list =[], ylabels :list =[], shape :tuple =(15, 15), guide =True, log=True) :
        plt.figure(figsize=shape)
        if title : plt.title(title)
        # If number is too big
        if log : 
            yyy= np.log(yyy) 
            ylabels=['log('+ylabel+")" for ylabel in ylabels]

        for i in range(len(xxx)) :
            # Adjust sub-figures
            ax = plt.subplot(2, len(xxx)//2+len(xxx)%2, i+1)
            if xlabels : plt.xlabel(xlabels[i], fontsize=shape[0])
            if ylabels : plt.ylabel(ylabels[i], fontsize=shape[1])
            # Plot the points
            xx, yy = xxx[i], yyy[i]
            plt.xticks(xx, xx)
            plt.yticks(yy, ["{:.3f}".format(y) for y in yy])
            main_line_style  = {'linestyle':'-',  'color':'k', 'linewidth':5}
            ax.plot(xx, yy, **main_line_style)
            # Appear the guide lines
            if guide :
                guide_line_style = {'linestyle':'--', 'color':'c', 'linewidth':1}
                for xinx in range(len(xx)) : ax.plot([xx[xinx]]*2, [min(yy), yy[xinx]], **guide_line_style)
                for yinx in range(len(yy)) : ax.plot([min(xx), xx[yinx]], [yy[yinx]]*2, **guide_line_style)
        plt.show()

class Show() :
    def __init__(self, stage :Stage) :
        self.stage = stage

    def seg4mask(self, segment, shape, color=(255,255,255), thickness=5) :
        mask = np.full(shape=shape, fill_value=0, dtype=np.uint8)
        poly = segment.reshape((-1, 2)).reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img=mask, pts=[poly], isClosed=True, color=color, thickness=thickness)
        return (mask).astype(np.uint8)

    # bbox should be [ x0, y0, x1, y1 ]
    def show4bbox(self, img, bbox, color=255) :
        if typc(img, "")   : img = cv2.imread(img)
        if typc(img, None) : log("There is no image"); return False
        img = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
        img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] = color
        self.stage.show(img, tool='plt')
    
    def show4cont(self, img, contours) :
        if typc(img, "")   : img = cv2.imread(img)
        if typc(img, None) : log("There is no image"); return False
        img = img + np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
        img = cv2.drawContours(img, contours, -1, (255,255,255), 10)
        self.stage.show(img, tool='plt')

    # segment should be [ x0, y0, ... , xn, yn ]
    def show4segs(self, img, segments) :
        if typc(img, "")   : img = cv2.imread(img)
        if typc(img, None) : log("There is no image"); return False
        cpimg = img.copy()
        for segment in segments : 
            if typc(segment, list()) : segment = np.array(segment, dtype=object)
            cpimg[self.seg4mask(segment, cpimg.shape) >= 255] = 0
        self.stage.show(cpimg, tool='plt')

    def compareMetrics(self, METC_PATH, step_size :int =5000, title :str ="", 
            xfields :list =['iteration'], yfields :list =['total_loss'], cry :bool =True, 
            log :bool =True, guide :bool =True) :
        metrics = load(join(METC_PATH, 'metrics.json'))
        metrics = list(filter(lambda metric:metric['iteration']!=0 and (metric['iteration']+1)%step_size==0, metrics))

        indice = {v:i for i,v in enumerate(metrics[0].keys())}
        matrix  = np.array([list(metric.values()) for metric in metrics])
 
        fxs = [list(matrix[:, indice[xfield]]) for xfield in xfields] 
        fys = [list(matrix[:, indice[yfield]]) for yfield in yfields]
        
        if cry : 
            if  len(fxs) == 1  : 
                self.stage.plot(fxs[0], fys[0], title=title, xlabel=xfields[0], ylabel=yfields[0], log=log, guide=guide)

            else    : 
                self.stage.plots(fxs, fys, title=title, xlabels=xfields, ylabels=yfields, log=log, guide=guide)
        
        mm = { yfields[i]: (fys[i].index(max(fys[i])), max(fys[i]), 
                            fys[i].index(min(fys[i])), min(fys[i])) for i in range(len(fys)) }
        return mm


if __name__=="__main__" :
    pass