import torch
from .BaseDataset import BaseDataset
from .augs import get_augs

def get_dataloader(params):

    train_augs, test_augs = get_augs(params)

    train_dataset = BaseDataset(params['img_folder'],
                                params['train_markup'],
                                train_augs,
                                train = True,
                                )
    params['num_classes'] = train_dataset.num_classes
    
    test_dataset = BaseDataset(params['img_folder'],
                               params['test_markup'],
                               test_augs
                               )

    train_loader = torch.utils.data.DataLoader(
                                                train_dataset,
                                                batch_size=params['batch_size'],
                                                pin_memory=False,
                                                drop_last=True,
                                                num_workers=params['num_workers'],
                                                persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
                                                test_dataset,
                                                batch_size=params['batch_size'],
                                                pin_memory=False,
                                                drop_last=True,
                                                num_workers=params['num_workers'],
                                                persistent_workers=True,
    )
    return train_loader, test_loader