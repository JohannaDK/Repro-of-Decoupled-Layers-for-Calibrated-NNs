import lightning.pytorch as pl
from torch import nn, optim
from torchmetrics.classification import Accuracy
import foolbox.attacks as fa
import warnings
from foolbox import PyTorchModel, accuracy, samples
from src.utils.focal_loss import FocalLoss


class lt_disc_models(pl.LightningModule):
    def __init__(self, model, num_classes, device="cpu", loss='ce', gammas=None, probs=None):
        super().__init__()
        self.model = model
        self._device=device
        self.loss = loss
        if loss not in ['ce', 'fl', 'fla']:
            raise NotImplemented("Loss has to be one of the following: ['ce', 'fl', 'fla']")
        if loss in ['fla', 'fl'] and (not gammas or not probs) and (not isinstance(gammas, list) or not isinstance(probs, list)):
            raise RuntimeError('gammas and probs have to be defined for focal loss')
        self.gammas = gammas
        self.probs = probs
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.model(x)

        if self.loss == 'ce':
            loss = nn.functional.cross_entropy(y_preds, y)
        elif self.loss == 'fl':
            loss = FocalLoss(device=self._device, gammas=self.gammas, probs=self.probs)(y_preds, y)
        elif self.loss == 'fla':
            loss = FocalLoss(device=self._device, gammas=self.gammas, probs=self.probs, adaptive=True)(y_preds, y)
        
        train_acc = self.train_acc(y_preds, y)
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.model(x)

        if self.loss == 'ce':
            loss = nn.functional.cross_entropy(y_preds, y)
        elif self.loss == 'fl':
            loss = FocalLoss(device=self._device, gammas=self.gammas, probs=self.probs)(y_preds, y)
        elif self.loss == 'fla':
            loss = FocalLoss(device=self._device, gammas=self.gammas, probs=self.probs, adaptive=True)(y_preds, y)

        val_acc= self.valid_acc(y_preds, y)
        self.log("valid_accuracy", val_acc, on_step=True, on_epoch=False)
        self.log("valid_loss", loss, on_step=True, on_epoch=False)
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)
        self.log('valid_acc_epoch', self.valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer