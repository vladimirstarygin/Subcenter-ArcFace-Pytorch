params = {

    'img_folder': '...',

    'train_markup': 'data_filtering/metadata/train.json',
    'test_markup':  'data_filtering/metadata/test.json',
    'markup_val': 'data_filtering/metadata/val.json',
    
    'checkpoint_dir': 'weights/data_filtration',

    #model config
    'backbone': 'resnet50',
    'embedding_dim': 2048,
    #autoset in get_loader training_script
    'num_classes': -1,

    #training params
    'batch_size': 160,
    'num_workers': 12,
    'im_size': 224,
    'max_epochs': 100,
    'save_freq': 5,

    #aug params
    'blur_limit': 10,
    'hue_shift': 5,
    'sat_shift': 40,
    'val_shift': 40,
    'mean': [0.491, 0.366, 0.29],
    'std': [0.25, 0.25, 0.22],

    #Optimizer
    'lr': 10e-4,
    'wd': 10e-5,
    

}
