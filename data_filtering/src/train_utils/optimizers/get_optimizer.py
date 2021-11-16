import torch

def get_optimizer(_config, model, loss=None):
    
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': loss.parameters()}],
                                    lr=_config['lr'], weight_decay=_config['wd'])

    return optimizer