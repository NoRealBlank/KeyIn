import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/fp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--inpaintor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--model', default='vgg', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

# dimensionality of delta
parser.add_argument('--seg_length', type=int, default=10, help='max offsets between keyframes, which is one more than the max number of frames between keyframes.')


opt = parser.parse_args()
if opt.model_dir != '':
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-inpaintor-rnn_layers=%d-lr=%.4f-g_dim=%d-last_frame_skip=%d-seg_length=%d%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.inpaintor_rnn_layers, opt.lr, opt.g_dim, opt.last_frame_skip, opt.seg_length, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)


# ---------------- models ----------------
import models.lstm as lstm_models

if opt.model_dir != '':
    embedder = saved_model['embedder']
    inpaintor = saved_model['inpaintor']
else:
    embedder = lstm_models.EmbeddingMLP(2*opt.g_dim + opt.seg_length, [2*opt.rnn_size]*4, opt.rnn_size)
    inpaintor = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.inpaintor_rnn_layers, opt.batch_size)
    embedder.apply(utils.init_weights)
    inpaintor.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
        
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

embedder_optimizer = opt.optimizer(embedder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
inpaintor_optimizer = opt.optimizer(inpaintor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


# --------- loss functions ------------------------------------
bce_criterion = nn.BCELoss()


# --------- transfer to gpu ------------------------------------
embedder.cuda()
inpaintor.cuda()
encoder.cuda()
decoder.cuda()
bce_criterion.cuda()


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()


# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 1 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    len_in = random.randint(3, 8)
    delta_key = []
    for i in range(opt.seg_length):
        if i == len_in-2:
            delta_key.append(0.8)
        else:
            delta_key.append(0.2/opt.seg_length)
    delta_key = torch.tensor(delta_key).unsqueeze(0).repeat(opt.batch_size, 1).cuda()

    h_seq = [encoder(x[i]) for i in range(len_in)]
    h_cond = torch.cat([h_seq[0][0], h_seq[len_in-1][0], delta_key], 1)

    for s in range(nsample):
        inpaintor.hidden = inpaintor.cond_hidden(embedder(h_cond))
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, len_in-1):
            if opt.last_frame_skip or i == 1:
                h, skip = h_seq[i-1]
                h = h.detach()

            h = inpaintor(h).detach()
            x_in = decoder([h, skip]).detach()
            gen_seq[s].append(x_in)
        
        gen_seq[s].append(x[len_in-1])


    to_plot = []
    gifs = [ [] for t in range(len_in) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(len_in):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        for s in range(nsample):
            row = []
            for t in range(len_in):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)

        for t in range(len_in):
            row = []
            row.append(gt_seq[t][i])
            for s in range(nsample):
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

    # fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
    # utils.save_gif(fname, gifs)


def plot_rec(x, epoch):

    len_in = random.randint(3, 8)
    delta_key = []
    for i in range(opt.seg_length):
        if i == len_in-2:
            delta_key.append(0.8)
        else:
            delta_key.append(0.2/opt.seg_length)
    delta_key = torch.tensor(delta_key).unsqueeze(0).repeat(opt.batch_size, 1).cuda()

    h_seq = [encoder(x[i]) for i in range(len_in)]
    h_cond = torch.cat([h_seq[0][0], h_seq[len_in-1][0], delta_key], 1)
    inpaintor.hidden = inpaintor.cond_hidden(embedder(h_cond))

    gen_seq = []
    gen_seq.append(x[0])
    for i in range(1, len_in-1):
        if opt.last_frame_skip or i == 1:
            h, skip = h_seq[i-1]
        else:
            h = h_seq[i-1][0]
        h = h.detach()

        h_pred = inpaintor(h).detach()
        x_pred = decoder([h_pred, skip]).detach()
        gen_seq.append(x_pred)
    gen_seq.append(x[len_in-1])
   

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(len_in):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x):
    embedder.zero_grad()
    inpaintor.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    len_in = random.randint(3, 8)

    delta_key = []
    for i in range(opt.seg_length):
        if i == len_in-2:
            delta_key.append(0.8)
        else:
            delta_key.append(0.2/opt.seg_length)
    delta_key = torch.tensor(delta_key).unsqueeze(0).repeat(opt.batch_size, 1).cuda()
    
    h_seq = [encoder(x[i]) for i in range(len_in)]
    
    """
    We condition the inpainting on both keyframe embeddings, κn−1 and κn, 
     as well as the temporal offset between the two, δn , 
     by passing these inputs through a multi-layer perceptron 
     that produces the initial state of the inpainting LSTM.
    """
    h_cond = torch.cat([h_seq[0][0], h_seq[len_in-1][0], delta_key], 1)
    inpaintor.hidden = inpaintor.cond_hidden(embedder(h_cond))

    bce = 0
    for i in range(1, len_in-1):
        if opt.last_frame_skip or i == 1:	
            h, skip = h_seq[i-1]
        else:
            h = h_seq[i-1][0]

        h_pred = inpaintor(h)
        x_pred = decoder([h_pred, skip])

        # need to change the loss according to the paper
        bce += bce_criterion(x_pred, x[i])

    loss = bce
    loss.backward()

    embedder_optimizer.step()
    inpaintor_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return bce.data.cpu().numpy()/(len_in-2)


# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    embedder.train()
    inpaintor.train()
    encoder.train()
    decoder.train()
    epoch_bce = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        bce = train(x)
        epoch_bce += bce

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] bce loss: %.5f (%d)' % (epoch, epoch_bce/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    embedder.eval()
    inpaintor.eval()
    encoder.eval()
    decoder.eval()
    x = next(testing_batch_generator)
    plot(x, epoch)
    plot_rec(x, epoch)

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'embedder': embedder,
        'inpaintor': inpaintor,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    
    if (epoch + 1) % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

