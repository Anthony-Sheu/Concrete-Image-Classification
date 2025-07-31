import os
import cv2
import torch
import torchvision.transforms as tf
import pytorch_lightning as pl
import torchvision.models
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

class NeuralNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained = True)

        # freeze pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        # unfreeze final classification layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim = 1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-3)

# loading data
class Data(pl.LightningDataModule):
    def __init__(self, dir="data", size = 32):
        super().__init__()
        self.data_dir = dir
        self.batch_size = size
        self.transform = tf.Compose([
            tf.RandomResizedCrop(size = (224, 224), antialias=True),
            tf.RandomHorizontalFlip(p = 0.5),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        self.train_data = ImageFolder(os.path.join(self.data_dir, 'train'), transform = self.transform)
        self.val_data = ImageFolder(os.path.join(self.data_dir, 'val'), transform = self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers = 8, pin_memory = True, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers = 8, pin_memory = True, batch_size = self.batch_size)


# load and visualize image
# image_neg = "Negative/00001.jpg"
# img = cv2.imread(image_neg)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# training
model = NeuralNet()
data_module = Data()
wandb_logger = WandbLogger(project = "anomaly-detection", name = "resnet18-run")
trainer = pl.Trainer(max_epochs = 5, logger = wandb_logger, devices = 1, accelerator = "gpu")
trainer.fit(model, data_module)



