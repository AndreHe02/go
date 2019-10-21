from torch import nn

class BaseModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, batch):
        board_feat = None
        if self.encoder is not None:
            board_feat = self.encoder(batch['board_state'])

        vocab_prob = self.decoder(batch['tok_in'], batch['lengths'], board_feat)
        return vocab_prob