import numpy as np
import argparse
from PIL import Image
import time

import chainer
from chainer import cuda, Variable, serializers
from net import *

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('input')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--maxsize', default=1024, type=int, help='The maximum side size of the picture')
args = parser.parse_args()

model = FastStyleNet()
serializers.load_npz(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

start = time.time()
image = Image.open(args.input)

# Convert the image so its sides are always smaller than the max size
max_size = args.maxsize

width = image.size[0]
height = image.size[1]

new_width = width
new_height = height

if width > height:
	if width > max_size:
		new_width = max_size
		reduction_percentage = new_width / float(width)
		new_height = int(float(height) * float(reduction_percentage))
else:
	if height > max_size:
		new_height = max_size
		reduction_percentage = new_height / float(height)
		new_width = int(float(width) * float(reduction_percentage))

image = image.resize((new_width, new_height), 2)
image =  xp.asarray(image.convert('RGB'), dtype=xp.float32).transpose(2, 0, 1)
image = image.reshape((1,) + image.shape)
x = Variable(image)

y = model(x)
result = cuda.to_cpu(y.data)

result = result.transpose(0, 2, 3, 1)
result = result.reshape((result.shape[1:]))
result = np.uint8(result)
print time.time() - start, 'sec'

Image.fromarray(result).save(args.out)
