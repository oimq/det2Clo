from det2Clo import clo2Imagine as c2i, clo2Trainer as c2t
from os.path import join

if __name__=="__main__" :
    DATA_PATH = "/home/denny/datasets/iMaterialist/"
    TRAN_PATH = join(DATA_PATH, "train/")
    ANNO_PATH = join(DATA_PATH, "train.csv")
    LABL_PATH = join(DATA_PATH, "label_descriptions.json")
    CONF_PATH = join(DATA_PATH, 'conf/train_config.json')
    YAML_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
    
    CLASSES = [13, 16, 18, 19, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 36, 39, 40, 41, 42, 43, 44, 45]

    # ! Make factory
    factory = c2t.Factory()

    # ! Create categories and save that
    CATE_PATH = join(DATA_PATH, 'train_categories.json')
    # categories = factory.categories_imat(LABL_PATH, CLASSES)
    # save(CATE_PATH, categories)

    # ! Make manager after save categories
    manager = c2t.Manager(DATA_PATH, YAML_NAME, CONF_PATH)
    manager.load_categories(CATE_PATH)

    # ! Create datasets and save that
    SETS_PATH = join(DATA_PATH, 'train_datasets.json')
    # datasets = factory.datasets4imat(TRAN_PATH, ANNO_PATH, LABL_PATH, manager.categories, classes=CLASSES)
    # save(SETS_PATH, datasets)

    # ! Load the datasets
    manager.load_datasets(SETS_PATH)

    # # ! Make trainer after load categories and datasets
    trainer = c2t.Trainer(manager)
    trainer.training(BATCH_SIZE_PER_IMAGE=32)

    print("done")