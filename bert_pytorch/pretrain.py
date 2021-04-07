import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import numpy as np

from model import BERT, BYOL
from dataset import BERTDataset
from transformers import BertTokenizer
import pytorch_lightning as pl

from transformers import AdamW, get_linear_schedule_with_warmup



class BERTTrainer(pl.LightningModule):
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, proj_size, seq_len,
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
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.current_step = 1
        # cuda_condition = torch.cuda.is_available()
        # self.device = torch.device("cuda" if cuda_condition else "cpu")
        # self.device = "cuda"

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BYOL(bert, sequence_size=seq_len, projection_size=proj_size)
        # SelfSupervisedLearner(
        #     bert,
        #     image_size = ,
        #     hidden_layer = 'avgpool',
        #     projection_size = 256,
        #     projection_hidden_size = 4096,
        #     moving_average_decay = 0.99
        # )

        # Setting the Adam optimizer with hyper-param
        self.log_freq = log_freq

        self.example_input_array = (torch.randint(0, 1000, (128, 2, 128)), torch.randint(0, 1, (128, 2, 128)))

        self.save_hyperparameters()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def get_lr(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    def lr_foo(self, _):
        return np.min([
            np.power(self.current_step, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_step])

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, _):
        if self.global_step == 0:
            aa = self.example_input_array[0].to(self.device)
            bb = self.example_input_array[1].to(self.device)
            self.logger.experiment.add_graph(self.model, (aa, bb))

        x, y = batch

        loss = self.forward(x, y)
        self.current_step += 1
        self.logger.experiment.add_scalar('train_loss', loss.detach().item(), self.global_step)
        self.logger.experiment.add_scalar('lr', self.get_lr(), self.global_step)
        # print(self.global_step)
        # if self.global_step % 10 == 0:
        #     gc.collect()
        return {'loss': loss}

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = AdamW(trainable_parameters, lr=self.lr, eps=1e-8)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=self.lr_foo
        )
        return [optimizer], [{
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
            }]

    def on_before_zero_grad(self, _):
        if self.model.use_momentum:
            self.model.update_moving_average()


class InputMonitor(pl.Callback):
    def on_train_batch_start(self, pl_trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 == 0:
            x, y = batch
            sample_input = x[:, 0]
            sample_output = pl_module.model.online_encoder.get_representation(sample_input.to(pl_module.device))[0]
            pl_logger = pl_trainer.logger
            pl_logger.experiment.add_histogram("input", x, global_step=pl_trainer.global_step)
            pl_logger.experiment.add_histogram("mask", y, global_step=pl_trainer.global_step)
            pl_logger.experiment.add_histogram("repr_cls", sample_output[0, :], global_step=pl_trainer.global_step)
            pl_logger.experiment.add_histogram("repr_first", sample_output[1, :], global_step=pl_trainer.global_step)

    # def on_train_start(self, trainer, pl_model):
    #     n = 0
    #
    #     example_input1, example_input2 = pl_model.example_input_array
    #     example_input1.requires_grad = True
    #
    #     pl_model.zero_grad()
    #     output = pl_model(example_input1.to(pl_model.device), example_input2.to(pl_model.device))
    #     output[n].abs().sum().backward()
    #
    #     zero_grad_inds = list(range(example_input1.size(0)))
    #     zero_grad_inds.pop(n)
    #
    #     if example_input1.grad[zero_grad_inds].abs().sum().item() > 0:
    #         raise RuntimeError("Your model mixes data across the batch dimension!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", type=str, required=True, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=768, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")
    parser.add_argument("-p", "--proj_size", type=int, default=256, help="byol projection size")

    parser.add_argument("-b", "--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")
    parser.add_argument("-ac", "--accumulate", type=int, default=16, help="batch accumulate num")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0004, help="weight_decay of adam")
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
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    args.lr = args.lr * args.batch_size * args.accumulate / 256

    print("Creating BERT Trainer")
    model = BERTTrainer(bert, proj_size=args.proj_size, seq_len=args.seq_len,
                        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                        log_freq=args.log_freq)

    logger = pl.loggers.TensorBoardLogger('log', name='byol_mask')

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        default_root_dir=args.output_path,
        # log_save_interval=args.log_freq,
        accelerator='ddp',
        logger=logger,
        callbacks=[InputMonitor()]
    )

    print("Training Start")
    trainer.fit(model, train_data_loader)
    # for epoch in range(args.epochs):
    #     trainer.train(epoch)
    #     trainer.save(epoch, args.output_path)
    #
    #     if test_data_loader is not None:
    #         trainer.test(epoch)
