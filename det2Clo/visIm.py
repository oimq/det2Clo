from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer, ColorMode

class visIm() :
    def __init__(self) :
        self.SHOWTYPE = ['plt', 'cv2']
        pass

    def visualize(self, img, show_type) :
        if   show_type == self.SHOWTYPE[0] :
            if img.shape[2] == 3 : img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(img)
            plt.show()

        elif show_type == self.SHOWTYPE[1] :
            cv2.imshow(winname='Visualize a image', mat=img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def plot(self, xx, yy, xlabel=None, ylabel=None, guide=True) :
        figshape=(15, 15)
        plt.figure(figsize=figshape)
        plt.xticks(xx, xx)
        plt.yticks(yy, ["{:.3f}".format(y) for y in yy])
        if xlabel : plt.xlabel(xlabel, fontsize=figshape[0])
        if ylabel : plt.ylabel(ylabel, fontsize=figshape[1])
        main_line_style  = {'linestyle':'-',  'color':'k', 'linewidth':5}
        plt.plot(xx, yy, **main_line_style)
        guide_line_style = {'linestyle':'--', 'color':'c', 'linewidth':1}
        if guide :
            for xinx in range(len(xx)) : plt.plot([xx[xinx]]*2, [min(yy), yy[xinx]], **guide_line_style)
            for yinx in range(len(yy)) : plt.plot([min(xx), xx[yinx]], [yy[yinx]]*2, **guide_line_style)
        plt.show()

    def plots(self, xxx, yyy, xlabels=None, ylabels=None, guide=True, log=True) :
        figshape=(15, 15)
        plt.figure(figsize=figshape)
        if log : yyy= np.log(yyy); ylabels=['log('+ylabel+")" for ylabel in ylabels]
        for i in range(len(xxx)) :
            xx, yy = xxx[i], yyy[i]
            ax = plt.subplot(2, len(xxx)//2+len(xxx)%2, i+1)
            plt.xticks(xx, xx)
            plt.yticks(yy, ["{:.3f}".format(y) for y in yy])
            if xlabels : plt.xlabel(xlabels[i], fontsize=figshape[0])
            if ylabels : plt.ylabel(ylabels[i], fontsize=figshape[1])
            main_line_style  = {'linestyle':'-',  'color':'k', 'linewidth':5}
            ax.plot(xx, yy, **main_line_style)
            guide_line_style = {'linestyle':'--', 'color':'c', 'linewidth':1}
            if guide :
                for xinx in range(len(xx)) : ax.plot([xx[xinx]]*2, [min(yy), yy[xinx]], **guide_line_style)
                for yinx in range(len(yy)) : ax.plot([min(xx), xx[yinx]], [yy[yinx]]*2, **guide_line_style)
        plt.show()

    def show(self, img, show_type='plt', detects=None) :
        if show_type not in self.SHOWTYPE : return print("There are no type for [ {} ]".format(show_type))

        if detects :
            if type(img) == type("") : img = cv2.imread(img)
            if set(detects.keys())&set(['outputs', 'scale', 'cm', 'metadata']) :
                visual = Visualizer(
                    img[:, :, ::-1], 
                    metadata    = detects['metadata'], 
                    scale       = detects['scale'],
                )
                visual = visual.draw_instance_predictions(detects['outputs']['instances'].to('cpu'))
                self.visualize(visual.get_image()[:, :, ::-1], show_type='plt')

            else : print("detects must have {}. but your is {}".format(['outputs', 'scale', 'cm'], detects.keys()))
        
        else       :
            if type(img) == type("") :
                img = cv2.imread(img)
                if type(img) == type(None) : return print("There are no image from [ {} ].".format(img))
                self.visualize(img, show_type)
                
            elif type(img) == type(np.zeros(shape=(1,1))) :
                self.visualize(img, show_type)

            else : return print("There are no image type for [ {} ]".format(type(img)))