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
        return out[:, -1, :].squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
            x = torch.cat((x[:, 1:, :], pred.unsqueeze(0)), dim=1)

        return preds


class RNNModel(lightning.LightningModule):
    def __init__(self, input_size, output_size, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.lstm = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out[:, -1, :].squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
            x = torch.cat((x[:, 1:, :], pred.unsqueeze(0)), dim=1)

        return preds



class TransformerModel(lightning.LightningModule):
    START_TOKEN = (0.,0.,0.,0.)

    def __init__(self, input_size, learning_rate, num_encoder_layers, num_decoder_layers, dim_feedforward, nhead, seq_len):
        super().__init__()
        self.lr = learning_rate
        self.transformer = nn.Transformer(d_model = input_size, batch_first=True, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, nhead=nhead)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, src, tgt):
        tgt_len = tgt.shape[1]
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(src.device)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x, y)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x, y)
        out = out.reshape(y.shape)
        loss = self.criterion(out, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
                pred = self.forward(x, y)
                preds.append(pred.detach().numpy())
                true.append(y.detach().numpy())
            except StopIteration:
                break
        return np.array(preds), np.array(true)

    def autoregressive_predict(self, x, target_length):
        preds = []
        tgt = torch.tensor([self.START_TOKEN]).unsqueeze(0)
        self.eval()

        for _ in range(target_length):
            pred = self.forward(x, tgt)
            tgt = pred
            preds.append(pred.detach().numpy())
            # x = torch.cat((x[:, 1:, :], pred.unsqueeze(0)), dim=1)

        return preds
