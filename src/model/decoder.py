import torch
from torch import nn

class BaseDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size=300, num_layers=2, hidden_dim=256, dropout=0.5, board_state_dim=256, eps=1e-5):
        super(BaseDecoder, self).__init__()
        self.wemb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(input_size=emb_size + board_state_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out_vocab_layer = nn.Linear(hidden_dim, vocab_size)
        self.sm = nn.Softmax(dim=-1)
        self.board_state_dim = board_state_dim
        self.eps = eps

    def forward(self, tok_in, lengths, board_state):
        bsize, sent_length = tok_in.shape
        emb = self.wemb(tok_in)
        if self.board_state_dim != 0:
            board_state_feat = board_state.unsqueeze(1).repeat(1, sent_length, 1)
            emb = torch.cat((emb, board_state_feat), dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.rnn(packed)
        hiddens, lengths = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
        vocab_prob = self.sm(self.out_vocab_layer(hiddens)) + self.eps
        vocab_prob = torch.log(vocab_prob)
        return vocab_prob
