import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import numpy as np

import tqdm

from ..model import BERT, BYOL
from ..dataset import BERTDataset
from transformers import BertTokenizer
import pytorch_lightning as pl


class BERTTrainer(pl.LightningModule):
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, proj_size,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 log_freq: int = 10):
        super().__init__()
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        self.warmup_steps = warmup_steps
        self.current_step = 1
        # cuda_condition = torch.cuda.is_available()
        # self.device = torch.device("cuda" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BYOL(bert, proj_size, bert.hidden)
        # SelfSupervisedLearner(
        #     bert,
        #     image_size = ,
        #     hidden_layer = 'avgpool',
        #     projection_size = 256,
        #     projection_hidden_size = 4096,
        #     moving_average_decay = 0.99
        # )

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def lr_foo(self, _):
        return np.min([
            np.power(self.current_step, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_step])

    def forward(self, x):
        return self.model(x)

    def training_step(self, tokens, _):
        if self.global_step == 1:
            aa = torch.randint(0, 1000, (3, 2, 128))
            self.logger.experiment.add_graph(self.model, aa)

        loss = self.forward(tokens)
        self.current_step += 1
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        scheduler = LambdaLR(
            self.optim,
            lr_lambda=self.lr_foo
        )
        return [self.optim], [{
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
            }]

    def on_before_zero_grad(self, _):
        if self.model.use_momentum:
            self.model.update_moving_average()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")
    parser.add_argument("-p", "--proj_size", type=int, default=128, help="byol projection size")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    # print("Loading Vocab", args.vocab_path)
    # vocab = WordVocab.load_vocab(args.vocab_path)
    vocab = BertTokenizer.from_pretrained('bert-base-uncased').get_vocab()
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    model = BERTTrainer(bert, proj_size=args.proj_size,
                        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                        log_freq=args.log_freq)

    logger = pl.loggers.TensorBoardLogger('../log', name='byol_test')

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        default_root_dir=args.output_path,
        # log_save_interval=args.log_freq,
        accelerator='ddp',
        logger=logger
    )

    print("Training Start")
    trainer.fit(model, train_data_loader)
    # for epoch in range(args.epochs):
    #     trainer.train(epoch)
    #     trainer.save(epoch, args.output_path)
    #
    #     if test_data_loader is not None:
    #         trainer.test(epoch)
