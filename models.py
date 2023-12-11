import lightning
import torch
import torch.nn as nn
import numpy as np
import math

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



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(lightning.LightningModule):
    START_TOKEN = (0.,0.,0.,0.)

    def __init__(self, input_size, learning_rate, num_encoder_layers, num_decoder_layers, dim_feedforward, nhead, seq_len):
        super().__init__()
        self.lr = learning_rate
        self.pos_encoder = PositionalEncoding(input_size, dropout=0.1)
        self.transformer = nn.Transformer(d_model = input_size, batch_first=True, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, nhead=nhead)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, src, tgt):
        tgt_len = tgt.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)

        out = self.pos_encoder(src)
        out = self.transformer(src=out, tgt=tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
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

    def autoregressive_predict(self, x, tgt, target_length):
        preds = []
        self.eval()

        for _ in range(target_length):
            pred = self.forward(x, tgt)
            preds.append(pred.detach().numpy())
            tgt = torch.cat((tgt, pred[:,-1,:].unsqueeze(1)), dim=1)

        return pred[-1].detach().numpy()
