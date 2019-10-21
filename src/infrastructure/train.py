import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim

class FlattenedLoss(nn.Module):

    def __init__(self, reduction):
        super(FlattenedLoss, self).__init__()
        self.loss = torch.nn.NLLLoss(ignore_index=0, reduction=reduction)

    def forward(self, pred, target):
        bsize, sent_len, vocab_size = pred.shape
        pred = pred.reshape((-1, vocab_size))
        target = target.reshape((bsize * sent_len,))
        return self.loss(pred, target)


class Perplexity_trainer:

    def __init__(self, model, train_generator, eval_generator_init, exp_name):
        self.logdir = 'run_logs/' + exp_name
        self.logger = SummaryWriter(self.logdir, flush_secs=30)
        self.loss = FlattenedLoss(reduction='mean')
        self.eval_loss = FlattenedLoss(reduction='sum')
        self.model, self.train_generator, self.eval_generator_init = model, train_generator, eval_generator_init
        self.step_counter = 0
        self.optim = optim.Adam(self.model.parameters())

    def train_one_step(self, batch):
        self.model.train()
        print('training for step %d.' % self.step_counter)
        vocab_logprob = self.model(batch)
        perplexity = self.loss(vocab_logprob, batch['tok_out'])
        perplexity.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.logger.add_scalar('Loss/train', perplexity.data.item(), self.step_counter)
        self.step_counter += 1

    def eval(self):
        self.model.eval()
        eval_generator = self.eval_generator_init()
        loss = 0
        for batch_idx, batch in enumerate(eval_generator):
            print('evaluating batch %d.' % batch_idx)
            tok_out = self.model(batch)
            loss += self.eval_loss(tok_out, batch['tok_out']).data.item()
        return loss

    def train(self, train_step, eval_every):
        for train_idx in range(train_step):
            batch = next(self.train_generator)
            self.train_one_step(batch)
            if train_idx % eval_every == 0 and train_idx != 0:
                eval_idx = train_idx // eval_every
                print('evaluation round %d.' % eval_idx)
                loss = self.eval()
                self.logger.add_scalar('Loss/eval', loss, eval_idx)