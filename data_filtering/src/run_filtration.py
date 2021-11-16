import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.neptune import NeptuneLogger

from .dataprocessing.get_loader import get_dataloader
from .credentials import API_TOKEN

from .models.FilterModel import Filtermodel
from .train_utils.losses.get_loss import get_loss
from .train_utils.LiTtrainer import LightningFilterModule

from .configs.subcenter_config import params


neptune_logger = NeptuneLogger(
    api_key=API_TOKEN,
    project_name="",
    close_after_fit=False,
    params = params,
    experiment_name=""
)


if __name__ == '__main__':

    train_loader, test_loader = get_dataloader(params)
    clf_loss, metric_loss = get_loss(params)
    
    training_model = Filtermodel(n_classes=params['num_classes'],
                                embedding_dim=params['embedding_dim'],
                                backbone=params['backbone'],
                                pseudolabels=False)


    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=params['checkpoint_dir'],
                                                    filename = 'filter_train--{epoch:02d}-{val_loss:.2f}',
                                                    every_n_epochs=params['save_freq'],
                                                    save_weights_only=True)

    model = LightningFilterModule(training_model,
                                  clf_loss,
                                  metric_loss,
                                  params['lr'], params['wd'])

    trainer = Trainer(max_epochs=params['max_epochs'],
                    logger=neptune_logger,
                    callbacks=[model_checkpoint],
                    deterministic=True,
                    gpus=2,
                    terminate_on_nan = True
                    )

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)
