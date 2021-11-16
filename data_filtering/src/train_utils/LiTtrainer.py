import torch
import torchmetrics 
import numpy as np
import pytorch_lightning as pl

class LightningFilterModule(pl.LightningModule):

    def __init__(self,
                 model,
                 clf_loss,
                 metric_loss,
                 lr, wd
                      ):
        super(LightningFilterModule, self).__init__()
        self.model = model
        self.metric_loss = metric_loss
        self.clf_loss = clf_loss
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.test_F1 = torchmetrics.F1()
        self.lr = lr
        self.wd = wd

    def forward(self, x):
        return self.model(x) 

    def training_step(self, batch, batch_nb):
        image, labels = batch
        features = self(image)
        logits = self.metric_loss(features, labels)
        loss = self.clf_loss(logits,labels)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = np.mean(torch.Tensor([i['loss'] for i in training_step_outputs]).cpu().numpy())
        self.logger.experiment.log_metric('train_loss',avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.metric_loss.parameters()}],
                                        lr=self.lr, weight_decay=self.wd)
        return optimizer

    def validation_step(self, batch, batch_nb):
        image, labels = batch
        features = self(image)
        logits = self.metric_loss(features, labels)
        test_acc = self.test_acc(logits,labels)
        test_F1 = self.test_F1(logits,labels)
        return {'test_loss': self.clf_loss(logits, labels), 'test_F1': test_F1, 'test_acc': test_acc}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = np.mean(torch.Tensor([i['test_loss'] for i in validation_step_outputs]).cpu().numpy())
        test_acc = np.mean(torch.Tensor([i['test_acc'] for i in validation_step_outputs]).cpu().numpy())
        test_F1 = np.mean(torch.Tensor([i['test_F1'] for i in validation_step_outputs]).cpu().numpy())
        self.logger.experiment.log_metric('test_accuracy',test_acc)
        self.logger.experiment.log_metric('test_F1',test_F1)
        self.logger.experiment.log_metric('test_loss',avg_loss)