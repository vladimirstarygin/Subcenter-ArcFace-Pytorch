import albumentations as A
import albumentations.pytorch

def get_augs(params):
    """Function for returning augmentations from albumentations library"""
    h, w = params['im_size'], params['im_size']
    train_augs, test_augs = [], []

    train_augs.append(A.Blur(params['blur_limit']))


    train_augs.append(A.RandomBrightnessContrast())
    train_augs.append(A.HueSaturationValue(hue_shift_limit=params['hue_shift'],
                                           sat_shift_limit=params['sat_shift'],
                                           val_shift_limit=params['val_shift']))


    train_augs.append(A.Resize(h, w))
    train_augs.append(A.RandomRotate90())
    
    train_augs.append(A.Normalize(mean=params['mean'], std=params['std']))

    train_augs.append(albumentations.pytorch.transforms.ToTensorV2())

    test_augs = A.Compose(test_augs + [A.Resize(h, w), 
                                       A.Normalize(mean=params['mean'], std=params['std']),
                                       albumentations.pytorch.transforms.ToTensorV2()])
    return A.Compose(train_augs), test_augs
