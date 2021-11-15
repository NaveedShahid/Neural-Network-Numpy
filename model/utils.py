import numpy as np

def get_smoothing(sum, mu, gamma):
    if sum is None:
        return mu
    return gamma * sum + (1. - gamma) * mu


def get_im2col(x_shape, field_height, field_width, padding=1,
                       stride=1):
    # First figure out what the size of the output should be
    _, C, H, W = x_shape
    assert((H + padding - field_height) % stride == 0)
    assert((W + padding - field_height) % stride == 0)
    out_height = int((H + padding - field_height) // stride + 1)
    out_width = int((W + padding - field_width) // stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    pad_forward = padding // 2
    pad_backward = padding // 2 + (padding % 2 != 0)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_forward, pad_backward), (pad_forward, pad_backward)),
                      mode='constant')

    k, i, j = get_im2col(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    pad_forward = padding // 2
    pad_backward = padding // 2 + (padding % 2 != 0)
    N, C, H, W = x_shape
    H_padded, W_padded = H + padding, W + padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, pad_forward: -pad_backward, pad_forward: -pad_backward]

def generator(dataset, batch_size, shuffle=True):
    length = len(dataset)
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    start_index = 0
    while start_index < length:
        if type(dataset) is list or type(dataset) is tuple:
            batch = []
            for x in dataset:
                batch.append(x[indices[start_index: start_index + batch_size]])
            batch = tuple(batch)
        else:
            batch = dataset[indices[start_index: start_index + batch_size]]
        start_index += batch_size
        yield batch

class Dataset:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        length = self.y.shape[0]
        indices = np.arange(length)
        if self.shuffle:
            np.random.shuffle(indices)
        start = 0
        while start < length:
            batch_X = self.X[indices[start: start + self.batch_size]]
            batch_y = self.y[indices[start: start + self.batch_size]]
            start += self.batch_size
            yield batch_X, batch_y

    def __len__(self):
        length = self.y.shape[0]
        num_batch = length // self.batch_size + (length % self.batch_size != 0)
        return num_batch