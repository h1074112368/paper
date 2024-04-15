import os
import numpy as np
import  netCDF4 as nc
import pickle
import pandas as pd 
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # For tf.data and preprocessing only.
import keras
from keras import layers
from keras import ops
import  netCDF4 as nc
import os
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

all_epoches=10
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
config=tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
num_classes = 100
input_shape = (1040//2, 1600//2,2*7)

patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.05  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64*2  # Embedding dimension
num_mlp = 256  # MLP layer size
# Convert embedded patches to query, key, and values with a learnable additive
# value
qkv_bias = True
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = (1040, 1600,2*7)  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 4
num_epochs = 1
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = ops.reshape(
        x,
        (
            -1,
            patch_num_y,
            window_size,
            patch_num_x,
            window_size,
            channels,
        ),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = ops.reshape(
        windows,
        (
            -1,
            patch_num_y,
            patch_num_x,
            window_size,
            window_size,
            channels,
        ),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, height, width, channels))
    return x
class WindowAttention(layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = keras.Variable(
            initializer=relative_position_index,
            shape=relative_position_index.shape,
            dtype="int",
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = ops.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = ops.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = ops.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = ops.reshape(self.relative_position_index, (-1,))
        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0,
        )
        relative_position_bias = ops.reshape(
            relative_position_bias,
            (num_window_elements, num_window_elements, -1),
        )
        relative_position_bias = ops.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + ops.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = ops.cast(
                ops.expand_dims(ops.expand_dims(mask, axis=1), axis=0),
                "float32",
            )
            attn = ops.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = ops.reshape(attn, (-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = ops.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = ops.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv
class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = ops.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = ops.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = keras.Variable(
                initializer=attn_mask,
                shape=attn_mask.shape,
                dtype=attn_mask.dtype,
                trainable=False,
            )

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = ops.reshape(x, (-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = ops.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = ops.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = ops.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = ops.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = ops.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
# Using tf ops since it is only used in tf.data.
def patch_extract(images):
    # print(images.shape)
    batch_size = tf.shape(images)[0]
    # print(tf.shape(images))
    # print(batch_size)
    patches = tf.image.extract_patches(
        images=images,
        sizes=(1, 2, 2, 1),
        strides=(1, 2, 2, 1),
        rates=( 1, 1, 1,1),
        padding="VALID",
    )
    # print(patch_size[0], patch_size[1])
    patch_dim = patches.shape[-1]
    patch_num = patches.shape[1]
    # print(tf.shape(patches))
    # print(patches.shape)
    # print(batch_size, patch_num * patch_num, patch_dim)
    return tf.reshape(patches, (batch_size, patches.shape[1] * patches.shape[2], patch_dim))
# patch_extract(x_train[0])


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = ops.arange(start=0, stop=self.num_patch)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        x = ops.reshape(x, (-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = ops.concatenate((x0, x1, x2, x3), axis=-1)
        x = ops.reshape(x, (-1, (height//2//2 ) * (width//2//2 ),  C*4*4))
        # x = ops.reshape(x, (-1, 24*64,64,  C//(4)))
        return self.linear_trans(x)
def augment(x):
    x = tf.image.random_crop(x, size=(-1,image_dimension[0]//2, image_dimension[1]//2, image_dimension[2]))#3))
    x = tf.image.random_flip_left_right(x)
    return x
def reshape_data(x):
    print(x.shape)
    x=tf.reshape(x,(-1,image_dimension[0], image_dimension[1], image_dimension[2]))
    return x
def max_pool(x):
    print(x.shape)
    # x=tf.reshape(x,(1,image_dimension[0], image_dimension[1], image_dimension[2]))
    x=tf.nn.max_pool(x, ksize=[1,2,2, 1], strides=[1,2, 2, 1] , padding='SAME')
    # x=tf.nn.l2_normalize(x, dim = 3)
    return x
    # layers.MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), data_format='channels_last')
def load_pickle(name):
    
    with open(name,'rb') as f:
        return pickle.load(f)
# data_x=np.load("/mnt/data2/hyc/data_x.npz")
# data_y=np.load("/mnt/data2/hyc/data_y.npz")
data_x=np.ones((500,1040,1600,7,2))
data_y=np.ones((500,1040,1600,7,2))
index=list(range(len(data_x)))
random.shuffle(index)
len_index=len(index)
data_x=np.array(data_x)
train_index=index[int(0.1*len_index):]
test_index=index[:int(0.1*len_index)]

import os
if os.path.exists('/home/hyc/swin/swin/index_cldas.npy'):
    os.remove('/home/hyc/swin/swin/index_cldas.npy')
np.save('/home/hyc/swin/swin/index.npy',index)

num_train_index_samples = int(len(train_index) * (1 - validation_split))
num_val_index_samples = len(train_index) - num_train_index_samples
train_index, val_index = np.split(train_index, [num_train_index_samples])




def dataset_generator(indexes):
    for i in range(indexes):
        x=np.load(f"/mnt/data2/hyc/tfrecord/x_{'{:04d}'.format(i)}.npy")
        y=np.load(f"/mnt/data2/hyc/tfrecord/y_{'{:04d}'.format(i)}.npy")[:,:,1:4:2,:]
        yield x, y
dataset = (
    tf.data.Dataset.from_generator(
        dataset_generator,
        args=[train_index],
        output_types=(tf.float32, tf.float32),
        output_shapes=([1040, 1600,7,2], [1040, 1600,2,2]),
        )
    # tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(batch_size=batch_size)
    .map(lambda x, y: (reshape_data(x), y))
    .map(lambda x, y: (max_pool(x), y))
    # .map(lambda x, y: (augment(x), y))
    
    .map(lambda x, y: (patch_extract(x), y))
    .prefetch(tf.data.experimental.AUTOTUNE)
)


dataset_val = (
    tf.data.Dataset.from_generator(
        dataset_generator,
        args=[val_index],
        output_types=(tf.float32, tf.float32),
        output_shapes=([1040, 1600,7,2], [1040, 1600,2,2]),
        )
    .batch(batch_size=batch_size)
    .map(lambda x, y: (reshape_data(x), y))
    .map(lambda x, y: (max_pool(x), y))           
    .map(lambda x, y: (patch_extract(x), y))
        .prefetch(tf.data.experimental.AUTOTUNE)
)
dataset_test = (
    tf.data.Dataset.from_generator(
        dataset_generator,
        args=[test_index],
        output_types=(tf.float32, tf.float32),
        output_shapes=([1040, 1600,7,2], [1040, 1600,2,2]),
        )
    .batch(batch_size=batch_size)
    .map(lambda x, y: (reshape_data(x), y))
    .map(lambda x, y: (max_pool(x), y))           
    .map(lambda x, y: (patch_extract(x), y))
        .prefetch(tf.data.experimental.AUTOTUNE)
)

print()



