import lightning
import torch
import torch.nn as nn
import numpy as np

class LSTMModel(lightning.LightningModule):
    def __init__(self, input_size, output_size, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out[:,-1,:].squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict(self, dataloader):
        preds = []
        true = []
        it = iter(dataloader)

        self.eval()

        while True:
            try:
                x, y = next(it)
                pred = self.forward(x)
                preds.append(pred.detach().numpy())
                true.append(y.detach().numpy())
            except StopIteration:
                break
        return np.array(preds), np.array(true)

    def autoregressive_predict(self, x, target_length):
        preds = []
        self.eval()

        for _ in range(target_length):
            pred = self.forward(x)
            preds.append(pred.detach().numpy())
            x = torch.cat((x[:,1:,:], pred.unsqueeze(0)), dim=1)

        return preds
