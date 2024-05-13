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

parser.add_argument('--pred_model_dir', default='', help='base directory to save logs')
parser.add_argument('--inpt_model_dir', default='', help='inpaintor model directory')

parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')

parser.add_argument('--n_cond', type=int, default=5, help='number of frames to condition on')

parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--posterior_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--model', default='vgg', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

# dimensionality of delta
parser.add_argument('--seg_length', type=int, default=10, help='max offsets between keyframes, which is one more than the max number of frames between keyframes.')

# KeyIn
parser.add_argument('--pred_horizon', type=int, default=30, help='number of frames to process for discovering keyframes')

parser.add_argument('--keyframe_num', type=int, default=6, help='number of keyframes to predict')

parser.add_argument('--beta_kld', type=float, default=0.001, help='weighting on KL to prior')
parser.add_argument('--beta_inpaint', type=float, default=0.05, help='weighting on inpainting loss')


opt = parser.parse_args()

assert opt.inpt_model_dir != '', "inpt_model_dir must be specified!"
saved_inpaintor = torch.load('%s/model.pth' % opt.inpt_model_dir)['inpaintor']

if opt.pred_model_dir != '':
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_cond=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%d-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.n_cond, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
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
embedder = saved_inpaintor['embedder']
inpaintor = saved_inpaintor['inpaintor']
encoder = saved_inpaintor['encoder']
decoder = saved_inpaintor['decoder']

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

freeze_parameters(embedder)
freeze_parameters(inpaintor)
freeze_parameters(encoder)
freeze_parameters(decoder)


import models.lstm as lstm_models

if opt.key_model_dir != '':
    keyframe_conditioner = saved_model['keyframe_conditioner']
    keyframe_predictor = saved_model['keyframe_predictor']
    KeyValue_posterior = saved_model['KeyValue_posterior']
else:
    keyframe_conditioner = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    keyframe_predictor = lstm_models.keyframe_lstm(opt.z_dim, opt.g_dim, opt.seg_length, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    KeyValue_posterior = lstm_models.KeyValue_lstm(opt.g_dim, opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    keyframe_conditioner.apply(utils.init_weights)
    keyframe_predictor.apply(utils.init_weights)
    KeyValue_posterior.apply(utils.init_weights)

attention = lstm_models.KeyValueAttention(opt.g_dim, opt.z_dim)

def reparameterize(mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = Variable(logvar.data.new(logvar.size()).normal_())
    return eps.mul(logvar).add_(mu)

keyframe_conditioner_optimizer = opt.optimizer(keyframe_conditioner.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
keyframe_predictor_optimizer = opt.optimizer(keyframe_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
KeyValue_posterior_optimizer = opt.optimizer(KeyValue_posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


# --------- loss functions ------------------------------------
bce_criterion = nn.BCELoss()

def kl_criterion(mu, logvar):
  # keep the batch dimension
  return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


# --------- transfer to gpu ------------------------------------
embedder.cuda()
inpaintor.cuda()
encoder.cuda()
decoder.cuda()

keyframe_conditioner.cuda()
keyframe_predictor.cuda()
KeyValue_posterior.cuda()
attention.cuda()
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
    nsample = 5 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    h_seq = [encoder(x[i]) for i in range(opt.n_past)]
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h_seq[i-1]
                h = h.detach()
            elif i < opt.n_past:
                h, _ = h_seq[i-1]
                h = h.detach()
            if i < opt.n_past:
                z_t, _, _ = posterior(h_seq[i][0])
                frame_predictor(torch.cat([h, z_t], 1)) 
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t = torch.cuda.FloatTensor(opt.batch_size, opt.z_dim).normal_()
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)


    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        for s in range(nsample):
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for s in range(nsample):
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
    utils.save_gif(fname, gifs)


def plot_rec(x, epoch):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    h_seq = [encoder(x[i]) for i in range(opt.n_past+opt.n_future)]
    for i in range(1, opt.n_past+opt.n_future):
        h_target = h_seq[i][0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h_seq[i-1]
        else:
            h, _ = h_seq[i-1]
        h = h.detach()
        z_t, mu, logvar = posterior(h_target)
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)
   

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
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
    keyframe_conditioner.zero_grad()
    keyframe_predictor.zero_grad()
    KeyValue_posterior.zero_grad()

    # initialize the hidden state.
    keyframe_conditioner.hidden = keyframe_conditioner.init_hidden()
    keyframe_predictor.hidden = keyframe_predictor.init_hidden()
    KeyValue_posterior.hidden = KeyValue_posterior.init_hidden()

    h_seq = [encoder(x[i]) for i in range(max(opt.n_cond + opt.pred_horizon, opt.n_cond + opt.keyframe_num * opt.seg_length))]
    
    # to pallarelize the computation
    gen_seq = [[] for i in range(opt.keyframe_num + 1)]

    """
    To condition the keyframe predictor on past frames, 
     we initialize its state with the final state of another LSTM 
     that processes the conditioning frames.
    """
    for i in range(opt.n_cond):
        keyframe_conditioner(h_seq[i][0])
    keyframe_predictor.hidden = keyframe_conditioner.hidden
    
    # TODO: They didn't say how to condition the posterior on past frames.
    # KeyValue_posterior.hidden = keyframe_conditioner.hidden

    # inference the future keyframes
    key_set = [[] for i in range(opt.pred_horizon)]
    value_mu_set = [[] for i in range(opt.pred_horizon)]
    value_logvar_set = [[] for i in range(opt.pred_horizon)]
    for i in range(opt.n_cond, opt.n_cond + opt.pred_horizon):
        # inverse inference flow
        index = opt.n_cond + opt.n_cond + opt.pred_horizon - i - 1
        key, value_mu, value_logvar = KeyValue_posterior(h_seq[index][0])
        key_set[index - opt.n_cond] = key
        value_mu_set[index - opt.n_cond] = value_mu
        value_logvar_set[index - opt.n_cond] = value_logvar
    
    # change shape to [batch_size, num_keys, key_dim or value_dim]
    key_set.transpose(0, 1)
    value_mu_set.transpose(0, 1)
    value_logvar_set.transpose(0, 1)

    keyframes = [[] for i in range(opt.keyframe_num + 1)]
    keyframes[0] = x[opt.n_cond - 1]
    keyframes_embed = [[] for i in range(opt.keyframe_num + 1)]

    # TODO: think out how to set the skip connection?
    keyframes_embed[0], keyframe_skip = encoder(keyframes[0])

    delta_set = [[] for i in range(opt.keyframe_num + 1)]
    delta_set[0] = torch.zeros(opt.batch_size, opt.seg_length)

    # TODO: consider its size
    index_set = torch.full((opt.keyframe_num + 1, opt.batch_size), fill_value=opt.n_cond-1, dtype=torch.int)
    
    # generate keyframes and inpaint the frames between them
    kld_set = [torch.zeros(opt.batch_size)]
    for i in range(1, opt.keyframe_num + 1):
        # get the prior distribution of the latent variable
        mu = attention(keyframes_embed[i-1], key_set, value_mu_set)
        logvar = attention(keyframes_embed[i-1], key_set, value_logvar_set)
        z_t = reparameterize(mu, logvar)

        # predict the next keyframe
        keyframes_embed[i], delta_set[i] = keyframe_predictor(z_t)
        keyframes[i] = decoder([keyframes_embed[i], keyframe_skip])

        # TODO: consider the case where index_set[i] > horizon
        # just set it to the last frame？
        index_set[i] = index_set[i-1] + torch.argmax(delta_set[i], dim=1) + 1

        # loss for keyframe prior
        kld_set.append(kl_criterion(mu, logvar))

        # inpaint the frames between the keyframes
        h_cond = torch.cat([keyframes_embed[i-1], keyframes_embed[i], delta_set[i]], 1)
        inpaintor.hidden = inpaintor.condition(embedder(h_cond))

        # TODO: consider the index difference between the index_set[i-1] batch
        for j in range(opt.seg_length):
            if j == 0:
                h, skip = encoder(keyframes[i-1])

            h = inpaintor(h)
            x_pred = decoder([h, skip])
            gen_seq.append(x_pred)
    
    """
    Convert distributions of interframe offsets δ n to keyframe timesteps τ n
    """
    tau_set = torch.zeros(opt.keyframe_num + 1, opt.batch_size, opt.pred_horizon)
    tau_set[1, :, 0:opt.seg_length] = delta_set[1]
    for t in range(1, opt.pred_horizon):
        for n in range(2, opt.keyframe_num + 1):
            for j in range(min(opt.seg_length, t)):
                tau_set[n, :, t] += tau_set[n-1, :, t-1-j] * delta_set[n, :, j]
    
    """
    Compute probabilities of keyframes being within the predicted sequence
    """
    c_set = torch.zeros(opt.keyframe_num + 1, opt.batch_size)
    for n in range(1, opt.keyframe_num + 1):
        c_set[n] = torch.sum(tau_set[n], dim=1)

    """
    Compute soft keyframe targets
    """
    keyframe_target = [[] for i in opt.keyframe_num + 1]
    for n in range(1, opt.keyframe_num + 1):
        keyframe_target[n] = torch.einsum('ib...,bi->b...', x[opt.cond:opt.cond + opt.pred_horizon], tau_set[n])

    """
    Compute the keyframe loss
    """
    bce_keyframe = bce_criterion(keyframes, keyframe_target, weight=c_set.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    kld = torch.sum(kld_set * c_set) / torch.sum(c_set)

    """
    Get probabilities of segments ending after particular frames
    """
    p_end = torch.zeros(opt.keyframe_num + 1, opt.batch_size, opt.seg_length)
    for n in range(1, opt.keyframe_num + 1):
        for t in range(opt.seg_length):
            p_end[n, :, t] = torch.sum(delta_set[n, :, t:opt.seg_length], dim=1)
    
    """
    Get distributions of individual frames’ timesteps
    """
    p_frame = torch.zeros(opt.keyframe_num + 1, opt.batch_size, opt.pred_horizon, opt.seg_length)
    for n in range(1, opt.keyframe_num + 1):
        for t in range(opt.pred_horizon):
            for j in range(min(opt.seg_length, t)):
                p_frame[n, :, t, j] = tau_set[n-1, :, t-1-j] * p_end[n, :, j]

    """
    Compute soft individual frames
    """
    frame_soft = [[] for i in opt.horizon]
    for i in range(opt.pred_horizon):
        frame_soft[i] = torch.einsum('nbj,njb...->b...', p_frame[:, :, i, :], gen_seq)

    """
    Compute the inpainting loss
    """
    bce_inpaint = bce_criterion(frame_soft, x[opt.cond:opt.cond + opt.pred_horizon])

    """
    Compute total sequence loss
    """
    loss = bce_keyframe + opt.beta_kld * kld + opt.beta_inpaint * bce_inpaint

    loss.backward()

    keyframe_conditioner_optimizer.step()
    keyframe_predictor_optimizer.step()
    KeyValue_posterior_optimizer.step()

    return bce_keyframe.data.cpu().numpy(), kld.data.cpu().numpy(), bce_inpaint.data.cpu().numpy()


# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    keyframe_conditioner.train()
    keyframe_predictor.train()
    KeyValue_posterior.train()

    epoch_bce_keyframe = 0
    epoch_kld = 0
    epoch_bce_inpaint = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        # train the predictor
        bce_keyframe, kld, bce_inpaint = train(x)
        epoch_bce_keyframe += bce_keyframe
        epoch_kld += kld
        epoch_bce_inpaint += bce_inpaint

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] bce_keyframe loss: %.5f | kld loss: %.5f | bce_inpaint loss: %.5f (%d)' % (epoch, epoch_bce_keyframe/opt.epoch_size, epoch_kld/opt.epoch_size, epoch_bce_inpaint/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # plot some stuff
    keyframe_conditioner.eval()
    keyframe_predictor.eval()
    KeyValue_posterior.eval()
    x = next(testing_batch_generator)
    plot(x, epoch)
    plot_rec(x, epoch)

    # save the model
    torch.save({
        'keyframe_conditioner': keyframe_conditioner,
        'keyframe_predictor': keyframe_predictor,
        'KeyValue_posterior': KeyValue_posterior,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    
    if (epoch + 1) % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

