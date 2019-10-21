import sys
sys.path.append('.')
from model.board_encoder import BaseBoardEncoder
from model.decoder import BaseDecoder
from model.model import BaseModel
from loader.loader import BaseLoader
from infrastructure.train import Perplexity_trainer

debug = False
train_step = 1000000 if not debug else 100
eval_every = 5000 if not debug else 3
print_every = 1000 if not debug else 2

def w_encoder():
    train_generator, eval_laoder_init, train_loader = BaseLoader.obtain_generators(debug=debug)
    encoder_model = BaseBoardEncoder()
    decoder_model = BaseDecoder(train_loader.vocab_size, board_state_dim=encoder_model.out_dim)
    model = BaseModel(encoder_model, decoder_model)
    print(model)
    pt = Perplexity_trainer(model, train_generator, eval_laoder_init, 'test', print_every=print_every)
    pt.train(train_step=train_step, eval_every=eval_every)

def wout_encoder():
    train_generator, eval_laoder_init, train_loader = BaseLoader.obtain_generators(debug=debug)
    decoder_model = BaseDecoder(train_loader.vocab_size, board_state_dim=0)
    model = BaseModel(None, decoder_model)
    print(model)
    pt = Perplexity_trainer(model, train_generator, eval_laoder_init, 'test', print_every=print_every)
    pt.train(train_step=train_step, eval_every=eval_every)

if __name__ == '__main__':
    wout_encoder()
    w_encoder()
