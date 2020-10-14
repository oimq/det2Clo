from det2Clo import clo2Imagine as c2i

if __name__=="__main__" :
    stage = c2i.Stage()
    show = c2i.Show(stage)
    METC_PATH = "/home/denny/datasets/iMaterialist/outputs/"

    METC_TITLE = "Accuracy Matrix"
    METC_FIELDS_Y = [
        'fast_rcnn/cls_accuracy', 
        'mask_rcnn/accuracy', 
        'fast_rcnn/false_negative', 
        'mask_rcnn/false_negative'
    ]
    METC_FIELDS_X = ['iteration'] * len(METC_FIELDS_Y)
    mm = show.compareMetrics(METC_PATH, step_size=5000, title=METC_TITLE, xfields=METC_FIELDS_X, yfields=METC_FIELDS_Y, log=False, guide=True)

    METC_TITLE = "Loss Matrix"
    METC_FIELDS_Y = [
        'total_loss', 
        'loss_box_reg', 
        'loss_cls', 
        'loss_mask',
        'loss_rpn_cls', 
        'loss_rpn_loc',
    ]
    METC_FIELDS_X = ['iteration'] * len(METC_FIELDS_Y)
    show.compareMetrics(METC_PATH, step_size=5000, title=METC_TITLE, xfields=METC_FIELDS_X, yfields=METC_FIELDS_Y, log=False, guide=True)
