import argparse
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
from torchmetrics import Accuracy

from model import BERT, BYOL
from dataset import BERTDataset
from transformers import BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pretrain import BERTTrainer
import tqdm
import csv


class GLUEDataset(Dataset):
    def __init__(self, seq_len, task, phase):
        self.seq_len = seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.get_vocab()
        self.no_mask = [self.vocab["[CLS]"], self.vocab["[PAD]"]]
        with open("/data/data/glue_data/"+task+"/"+phase+".tsv") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            self.data = []
            self.label = []
            if task == "CoLA":
                for row in tqdm.tqdm(reader):
                    sentence = [self.vocab["[CLS]"]] + [self.vocab[i] for i in self.tokenizer.tokenize(row[3])] + [self.vocab["[SEP]"]]
                    sentence = sentence + [self.vocab["[PAD]"] for _ in range(seq_len - len(sentence))]
                    # self.data.append({"sentence": sentence[:seq_len], "label": row[1]})
                    self.data.append(sentence[:seq_len])
                    self.label.append(int(row[1]))
        print(self.label.count(1) / len(self.label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # output = {"bert_input": self.data[item],
        #           "label": self.label[item]}
        #
        # return {key: torch.tensor(value) for key, value in output.items()}
        # return {"input":self.data[item], "label":self.label[item]}
        # return {"sentence":self.data[item], "label":self.label[item]}
        mask_idx = np.array([1 if t not in self.no_mask else 0 for t in self.data[item]])
        ret_mask = torch.tensor(mask_idx, dtype=torch.int)
        del mask_idx
        return torch.tensor(self.data[item], dtype=torch.long), torch.tensor(self.label[item], dtype=torch.long), ret_mask


class FineTuner(pl.LightningModule):
    def __init__(self, model, label_size, lr: float = 1e-5, betas=(0.9, 0.999), weight_decay: float = 0.0004):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.model = model
        self.model.train()
        self.dense = nn.Linear(512, 512)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(0.1)
        self.logits = nn.Linear(512, 2)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        # self.save_hyperparameters()

    def get_representation(self, x):
        return self.byol.model.online_encoder.get_representation(x)[:, 0]

    def get_representation_words(self, x, mask):
        repr = self.model(x)
        output = torch.stack([r[torch.where(mask[idx] > 0)].mean(dim=0) for idx, r in enumerate(repr)])
        output = self.dense(output)
        output = self.activation(output)
        return output

    def get_lr(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    def forward(self, x, mask):
        # with torch.no_grad():
        pool = self.get_representation_words(x, mask)
        # print(output[0][:10])
        logits = self.logits(self.dropout(pool))
        print(logits.shape)
        return logits

    def loss(self, logits, labels):
        return self.criterion(logits.view(-1, 2), labels.view(-1))

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        # print(pred_flat)
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = AdamW(trainable_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=1000)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
        }]

    def training_step(self, batch, _):
        tokens, label, mask = batch
        # tokens, label = batch['sentence'], batch['label']
        # if self.global_step == 0:
        #     aa = torch.randint(0, 1000, (3, 2, 128)).to(self.device)
        #     self.logger.experiment.add_graph(self.model, aa)

        logits = self.forward(tokens, mask)
        loss = self.loss(logits, label)

        self.log('train_loss', loss.item(), on_step=True, prog_bar=True)
        self.log('train_acc', self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy()), on_step=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        tokens, label, mask = batch
        # tokens, label = batch['sentence'], batch['label']
        logits = self.forward(tokens, mask)
        loss = self.loss(logits, label)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy()), prog_bar=True)


class InputMonitor(pl.Callback):
    def on_train_batch_start(self, pl_trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx % 10 == 0:
            x, y, z = batch
            sample_output = pl_module.get_representation_words(x.to(pl_module.device), z)
            pl_logger = pl_trainer.logger
            pl_logger.experiment.add_histogram("repr", sample_output, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("mask", y, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("repr_cls", sample_output[0, :], global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("repr_first", sample_output[1, :], global_step=pl_trainer.global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=2, help="dataloader worker size")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-ch", "--checkpoint", required=True, type=str, help="filename")

    args = parser.parse_args()

    print("Loading Train Dataset")
    train_dataset = GLUEDataset(args.seq_len, "CoLA", "train")

    print("Loading Validation Dataset")
    val_dataset = GLUEDataset(args.seq_len, "CoLA", "dev")

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    bert = BERT(len(BertTokenizer.from_pretrained('bert-base-uncased').get_vocab()), hidden=512, n_layers=8, attn_heads=8)
    encoder = BERTTrainer.load_from_checkpoint(args.checkpoint, bert=bert, proj_size=256)

    model2 = FineTuner(encoder.model.net, 2)

    logger = pl.loggers.TensorBoardLogger('log', name="byol_finetune")

    tuner = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        default_root_dir=args.output_path,
        accelerator='ddp',
        val_check_interval=17,
        logger=logger,
        callbacks=[InputMonitor()]
    )

    print("Start!!")
    tuner.fit(model2, train_data_loader, val_data_loader)
