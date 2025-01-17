import os
import argparse
from PIL import Image

from chainer import cuda, Variable, optimizers, serializers
from net import *

def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

def total_variation_regularization(x, beta=1):
    xp = cuda.get_array_module(x.data)
    wh = Variable(xp.array([[[[1],[-1]],[[1],[-1]],[[1],[-1]]]], dtype=xp.float32))
    ww = Variable(xp.array([[[[1, -1]],[[1, -1]],[[1, -1]]]], dtype=xp.float32))
    tvh = lambda x: F.convolution_2d(x, W=wh, pad=1)
    tvw = lambda x: F.convolution_2d(x, W=ww, pad=1)

    dh = tvh(x)
    dw = tvw(x)
    tv = (F.sum(dh**2) + F.sum(dw**2)) ** (beta / 2.)
    return tv

parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', default='out', type=str,
                    help='output model file path without extension')
parser.add_argument('--lambda_tv', default=10e-4, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', default=1e0, type=float)
parser.add_argument('--lambda_style', default=1e1, type=float)
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
args = parser.parse_args()

batchsize = args.batchsize

n_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style
output = args.output
fs = os.listdir(args.dataset)
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
n_data = len(imagepaths)
print 'num traning images:', n_data
n_iter = n_data / batchsize
print n_iter, 'iterations,', n_epoch, 'epochs'

model = FastStyleNet()
vgg = VGG()
serializers.load_npz('vgg16.model', vgg)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    vgg.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

O = optimizers.Adam(alpha=args.lr)
O.setup(model)

style = vgg.preprocess(np.asarray(Image.open(args.style_image).convert('RGB').resize((512,512),2), dtype=np.float32))
style = xp.asarray(style, dtype=xp.float32)
style_b = xp.zeros((batchsize,) + style.shape, dtype=xp.float32)
for i in range(batchsize):
    style_b[i] = style
feature_s = vgg(Variable(style_b, volatile=True))
gram_s = [gram_matrix(y) for y in feature_s]

for epoch in range(n_epoch):
    print 'epoch', epoch
    for i in range(n_iter):
        model.zerograds()
        vgg.zerograds()

        indices = range(i * batchsize, (i+1) * batchsize)
        x = xp.zeros((batchsize, 3, 512, 512), dtype=xp.float32)
        for j in range(batchsize):
            x[j] = xp.asarray(Image.open(imagepaths[i*batchsize + j]).convert('RGB').resize((512,512),2), dtype=np.float32).transpose(2, 0, 1)

        xc = Variable(x.copy(), volatile=True)
        x = Variable(x)

        y = model(x)

        xc -= 120
        y -= 120

        feature = vgg(xc)
        feature_hat = vgg(y)

        L_feat = lambda_f * F.mean_squared_error(Variable(feature[2].data), feature_hat[2]) # compute for only the output of layer conv3_3

        L_style = Variable(xp.zeros((), dtype=np.float32))
        for f, f_hat, g_s in zip(feature, feature_hat, gram_s):
            L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), Variable(g_s.data))

        L_tv = lambda_tv * total_variation_regularization(y)
        L = L_feat + L_style + L_tv

        print '(epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter, L.data)

        L.backward()
        O.update()

        if args.checkpoint > 0 and i % args.checkpoint == 0:
            serializers.save_npz('models/{}_{}_{}.model'.format(output, epoch, i), model)

    print 'save "style.model"'
    serializers.save_npz('models/{}_{}.model'.format(output, epoch), model)

serializers.save_npz('models/{}.model'.format(output), model)
