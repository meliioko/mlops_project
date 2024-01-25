from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
import pytorch_lightning as pl
import numpy as np
from torchvision.transforms import v2
import torch

class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.val_losses = []
        self.val_accs = []
        self.epoch = 0
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(batch)
        return torch.argmax(out, dim=1).tolist()


    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self(images)
        loss = self.loss(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self(images)
        loss = self.loss(outputs, labels)

        labels_hat = torch.argmax(outputs, dim=1)
        val_acc = torch.sum(labels == labels_hat).item() / (len(labels) * 1.0)
        self.val_accs.append(val_acc)
        self.val_losses.append(loss.item())
       
      
    def on_validation_epoch_end(self):
        acc = np.mean(self.val_accs)
        print(f"Val acc:{acc}")
        loss = np.mean(self.val_losses)
        self.log(f'val_acc_{self.epoch}', acc)
        self.log(f'val_loss_{self.epoch}', loss)
        self.epoch += 1

    def test_step(self, batch, batch_idx):
      x, y = batch

      # implement your own
      out = self(x)
      loss = self.loss(out, y)


      # calculate acc
      labels_hat = torch.argmax(out, dim=1)
      test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

      # log the outputs!

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


def inference(images, model):
    labels = ['calling', 'clapping','cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music', 'running',
              'sitting', 'sleeping', 'texting', 'using_laptop']
    transform = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    data = DataLoader([transform(image) for image in images], batch_size=32, shuffle=True)
    trainer = pl.Trainer()
    outputs = trainer.predict(model, data)
    predicted_labels = []
    for label_index in outputs[0]:
        label = labels[label_index]
        predicted_labels.append(label)

    return predicted_labels