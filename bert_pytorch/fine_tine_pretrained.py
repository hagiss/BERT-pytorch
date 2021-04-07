from transformers import BertModel, AdamW, BertConfig

import torch
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import BertTokenizer
from torch import nn
import pytorch_lightning as pl
import tqdm
import csv
from torchmetrics import Accuracy


class GLUEDataset(Dataset):
    def __init__(self, task, phase, seq_len=256):
        self.seq_len = seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.get_vocab()
        self.no_mask = [self.vocab["[PAD]"]]
        with open("/data/data/glue_data/" + task + "/" + phase + ".tsv") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            self.data = []
            self.label = []
            if task == "CoLA":
                for row in tqdm.tqdm(reader):
                    sentence = [self.vocab["[CLS]"]] + [self.vocab[i] for i in self.tokenizer.tokenize(row[3])] + [
                        self.vocab["[SEP]"]]
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
        return torch.tensor(self.data[item], dtype=torch.long), np.array(self.label[item]), ret_mask


class InputMonitor(pl.Callback):
    def on_train_batch_start(self, pl_trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx % 10 == 0:
            x, y, z = batch
            sample_input = x
            sample_output = pl_module.model(sample_input.to(pl_module.device), z.to(pl_module.device)).pooler_output
            pl_logger = pl_trainer.logger
            # pl_logger.experiment.add_histogram("input", x, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("label", y, global_step=pl_trainer.global_step)
            pl_logger.experiment.add_histogram("repr_cls", sample_output, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("repr_first", sample_output[1, :], global_step=pl_trainer.global_step)


class Pretrained(pl.LightningModule):
    def __init__(self, model, seq_len=128, lr: float = 1e-5, betas=(0.9, 0.999), weight_decay: float = 0.01):
        super().__init__()

        self.model = model
        self.model.train()


        self.dropout = nn.Dropout(0.1)
        self.logits = nn.Linear(768, 2)
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.seq_len = seq_len
        self.betas = betas
        self.weight_decay = weight_decay

        self.num_labels = 2

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def forward(self, x, mask):
        repr = self.model(x, mask)[1]

        # print(output[0][:10])
        logits = self.logits(self.dropout(repr))
        return logits

    def loss(self, logits, labels):
        return self.criterion(logits.view(-1, self.num_labels), labels.view(-1))

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        print(pred_flat)
        # print(labels_flat)
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

        # logits, prob = self.forward(tokens)
        # loss = self.loss(logits, label)

        logits = self.forward(tokens, mask)
        loss = self.loss(logits, label)

        self.log('train_loss', loss.item(), on_step=True, prog_bar=True)
        self.log('train_acc', self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy()), on_step=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        tokens, label, mask = batch
    #     # tokens, label = batch['sentence'], batch['label']
    #     logits, prob = self.forward(tokens)
    #     self.logger.experiment.add_histogram("prob", prob, global_step=self.global_step)
    #     loss = self.loss(logits, label)
        logits = self.forward(tokens, mask)
        loss = self.loss(logits, label)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy()), prog_bar=True)


if __name__ == '__main__':

    print("Loading Train Dataset")
    train_dataset = GLUEDataset("CoLA", "train")

    print("Loading Validation Dataset")
    val_dataset = GLUEDataset("CoLA", "dev")

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=32, num_workers=1)

    bert = BertModel.from_pretrained("bert-base-uncased")
    # bert = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    #     num_labels=2,  # The number of output labels--2 for binary classification.
    #     You can increase this for multi-class tasks.
        # output_attentions=False,  # Whether the model returns attentions weights.
        # output_hidden_states=False,  # Whether the model returns all hidden-states.
    # )
    # Get all of the model's parameters as a list of tuples.

    model = Pretrained(bert)

    logger = pl.loggers.TensorBoardLogger('log', name="bert_pretrained")

    tuner = pl.Trainer(
        gpus=1,
        max_epochs=20,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        default_root_dir="output/bert.model",
        accelerator='ddp',
        val_check_interval=17,
        logger=logger,
        callbacks=[InputMonitor()]
    )

    print("Start!!")
    tuner.fit(model, train_data_loader, val_data_loader)
