from os.path import join
# from det2Clo.clo2Trainer import Factory, Manager, Trainer 
from det2Clo import clo2Trainer as c2t

if __name__=="__main__" :
    DATA_PATH = '/home/denny/datasets/deepfashion/'
    TRAN_PATH = join(DATA_PATH, 'train/')
    CONF_PATH = join(DATA_PATH, 'conf/train_config.json')
    LABL_PATH = join(DATA_PATH, 'train_descriptions.json')
    YAML_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
    
    # ! Make factory
    factory = c2t.Factory()
    
    # ! Create categories and save that
    CATE_PATH = join(DATA_PATH, 'train_categories.json')
    # categories = factory.categories_deep(TRAN_PATH=TRAN_PATH, sample_size=-1)
    # save(CATE_PATH, categories)

    # ! Make manager after save categories
    manager = c2t.Manager(DATA_PATH, YAML_NAME, CONF_PATH)
    manager.load_categories(CATE_PATH)

    # ! Create datasets and save that
    SETS_PATH = join(DATA_PATH, 'train_datasets.json')
    # datasets = factory.datasets4deep(TRAN_PATH, manager.categories)
    # save(SETS_PATH, datasets)
    
    manager.load_datasets(SETS_PATH)

    # ! Make trainer after load categories and datasets
    trainer = c2t.Trainer(manager)
    trainer.training(BATCH_SIZE_PER_IMAGE=32)

    print("done")