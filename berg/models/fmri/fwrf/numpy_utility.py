import numpy as np
import scipy.io as sio
from scipy.special import erf
import math

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 

def iterate_minibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs), batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]

        
def create_downsampling_vector(old_dim=1, new_dim=1, symmetric=True, preserve_norm=True):
    ratio = float(old_dim)/float(new_dim)
    x_range = np.arange(float(0), float(old_dim) + ratio/2, ratio)
    
    x_min = np.floor(x_range).astype(int)
    x_max = np.ceil(x_range).astype(int)
    x_min_frac = x_range - x_min
    x_max_frac = x_max - x_range
    stack = []
    for k,(xi,xf) in enumerate(zip(x_max[:-1], x_min[1:])):
        z = np.zeros(old_dim)
        z[xi:xf] = np.ones(xf-xi)
        if xi>0:
            z[xi-1] = x_max_frac[k]
        if xf<old_dim:
            z[xf] = x_min_frac[k+1]
        stack += [z,]
    # create a matrix that re-mix the entries
    if preserve_norm:
        return np.array(stack)
    else:
        stack = np.array(stack)
        return stack / np.sum(stack, axis=1, keepdims=True)

def create_upsampling_vector(old_dim=1, new_dim=1, symmetric=True, preserve_norm=True):
    ratio = float(new_dim)/float(old_dim-1)
    x_range = np.arange(float(0), float(new_dim) + ratio/2, ratio)
    stack = []
    for k,v in enumerate(np.arange(0,new_dim+1e-3,float(new_dim)/(new_dim-1))):
        z = np.zeros(old_dim)
        i = int(np.floor(v / ratio)) 
        if i+1<old_dim:
            d = (v - x_range[i]) / (x_range[i+1] - x_range[i])
            z[i], z[i+1] = 1-d, d
        else:
            z[i] = 1.0 
        stack += [z,]
    stack = np.array(stack)
    if preserve_norm:
        return stack / np.sum(stack, axis=0)
    else:
        return stack

def create_sampling_vector(old_dim=1, new_dim=1, symmetric=True, preserve_norm=True):
    if new_dim>old_dim:
        return create_upsampling_vector(old_dim, new_dim, symmetric, preserve_norm=preserve_norm) 
    else:
        return create_downsampling_vector(old_dim, new_dim, symmetric, preserve_norm=preserve_norm)
    
def create_sampling_array(old_dim=(1,1), new_dim=(1,1), symmetric=True, preserve_norm=True):
    ux = create_sampling_vector(old_dim[0], new_dim[0], symmetric=symmetric, preserve_norm=preserve_norm)
    uy = create_sampling_vector(old_dim[1], new_dim[1], symmetric=symmetric, preserve_norm=preserve_norm)
    stack = []
    for vx in np.array(ux):
        for vy in np.array(uy):
            stack += [np.outer(vx,vy),]
    return np.array(stack).reshape((len(stack), -1))


def make_gaussian(x, y, sigma, n_pix, size=None, dtype=np.float32):
    '''This will create a gaussian with respect to a standard coordinate system in which the center of the image is at (0,0) and the top-left corner correspond to (-size/2, size/2)'''
    deg = dtype(n_pix) if size==None else size
    dpix = dtype(deg) / n_pix
    pix_min = -deg/2. + 0.5 * dpix
    pix_max = deg/2.
    [Xm, Ym] = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix));
    d = (2*dtype(sigma)**2)
    A = dtype(1. / (d*np.pi))
    Zm = dpix**2 * A * np.exp(-((Xm-x)**2 + (-Ym-y)**2) / d)
    if(sigma<dpix/2):
        Zm /= np.sum(Zm)
    return Xm, -Ym, Zm.astype(dtype)

def make_gaussian_stack(xs, ys, sigmas, n_pix, size=None, dtype=np.float32):
    stack_size = min(len(xs), len(ys), len(sigmas))
    assert stack_size>0
    Z = np.ndarray(shape=(stack_size, n_pix, n_pix), dtype=dtype)
    X,Y,Z[0,:,:] = make_gaussian(xs[0], ys[0], sigmas[0], n_pix, size=size, dtype=dtype)
    for i in range(1,stack_size):
        _,_,Z[i,:,:] = make_gaussian(xs[i], ys[i], sigmas[i], n_pix, size=size, dtype=dtype)
    return X, Y, Z


def gaussian_mass(xi, yi, dx, dy, x, y, sigma):
    return 0.25*(erf((xi-x+dx/2)/(np.sqrt(2)*sigma)) - erf((xi-x-dx/2)/(np.sqrt(2)*sigma)))*(erf((yi-y+dy/2)/(np.sqrt(2)*sigma)) - erf((yi-y-dy/2)/(np.sqrt(2)*sigma)))
    
def make_gaussian_mass(x, y, sigma, n_pix, size=None, dtype=np.float32):
    deg = dtype(n_pix) if size==None else size
    dpix = dtype(deg) / n_pix
    pix_min = -deg/2. + 0.5 * dpix
    pix_max = deg/2.
    [Xm, Ym] = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix));
    if sigma<=0:
        Zm = np.zeros_like(Xm)
    elif sigma<dpix:
        g_mass = np.vectorize(lambda a, b: gaussian_mass(a, b, dpix, dpix, x, y, sigma)) 
        Zm = g_mass(Xm, -Ym)        
    else:
        d = (2*dtype(sigma)**2)
        A = dtype(1. / (d*np.pi))
        Zm = dpix**2 * A * np.exp(-((Xm-x)**2 + (-Ym-y)**2) / d)
    return Xm, -Ym, Zm.astype(dtype)   
    
def make_gaussian_mass_stack(xs, ys, sigmas, n_pix, size=None, dtype=np.float32):
    stack_size = min(len(xs), len(ys), len(sigmas))
    assert stack_size>0
    Z = np.ndarray(shape=(stack_size, n_pix, n_pix), dtype=dtype)
    X,Y,Z[0,:,:] = make_gaussian_mass(xs[0], ys[0], sigmas[0], n_pix, size=size, dtype=dtype)
    for i in range(1,stack_size):
        _,_,Z[i,:,:] = make_gaussian_mass(xs[i], ys[i], sigmas[i], n_pix, size=size, dtype=dtype)
    return X, Y, Z



def pruning_mask(shaped_as, prune_ratio=0.0):
    '''prune_ratio = 1. means discard everything'''
    return np.random.choice([True, False], size=shaped_as.shape[1], replace=True, p=[1.-prune_ratio, prune_ratio])#astype(np.bool)
    
def uniform_nsphere(batch, size):
    '''Returns a batch of uniformly distributed points on a nsphere'''
    nns = np.random.normal(0., 1., size=(batch, size)).astype(np.float32)
    nnss = np.sqrt(np.sum(np.square(nns), axis=1))
    return (nns.T / nnss).T

def uniform_ncube(batch, size):
    return np.random.uniform(-1., 1., size=(batch, size)).astype(np.float32) 

def normal_ncube(batch, size):
    return np.random.normal(0., 1., size=(batch, size)).astype(np.float32)



def sie(x, c=10): 
    '''SparseIntegerEmbedding'''
    y = np.zeros((len(x), c), dtype=np.float32)
    y[np.arange(len(x)), x] = 1
    return y

def place_tile_in(tile, new_npx):
    batch_size = tile.shape[0]
    features = tile.shape[1]
    A = np.zeros(shape=(batch_size, features, new_npx, new_npx), dtype=tile.dtype)
    dx = tile.shape[2]
    max_x = new_npx - dx
    pos_x = np.random.randint(0, max_x, size=batch_size)
    pos_y = np.random.randint(0, max_x, size=batch_size)
    for b in range(batch_size):
        A[b, :, pos_x[b]:pos_x[b]+dx, pos_y[b]:pos_y[b]+dx] = tile[b,...]
    return A

def mosaic_vis(X, pad=0, save_path=None):
    xmin, xmax = np.amin(X), np.amax(X)
    S = (X.astype(np.float32)-xmin)/(xmax-xmin)
    n = X.shape[0]
    x = int(np.ceil(np.sqrt(np.float32(n))))
    y = n // x
    while x*y<n:
        y+=1
    h, w = X.shape[1:3]
    if len(X.shape)==4:
        img = np.zeros((h*y+(y-1)*pad, w*x+(x-1)*pad, X.shape[3]))
        for k,s in enumerate(S):
            j, i = k//x, k%x
            img[j*pad+j*h:j*pad+j*h+h, i*pad+i*w:i*pad+i*w+w, :] = s    
    else:
        img = np.zeros((h*y+(y-1)*pad, w*x+(x-1)*pad))
        for k,s in enumerate(S):
            j, i = k//x, k%x
            img[j*pad+j*h:j*pad+j*h+h, i*pad+i*w:i*pad+i*w+w] = s
            
    if save_path is not None:
        imsave(save_path, img)
    return img


def select_along_axis(a, choice, run_axis=0, choice_axis=1):
    ''' run axis of lenght N
        choice axis of lenght M
        choice is a vector of lenght N with integer entries between 0 and M (exclusive).
        Equivalent to:
        >   for i in range(N):
        >       r[...,i] = a[...,i,...,choice[i],...]
        returns an array with the same shape as 'a' minus the choice_axis dimension
    '''
    assert len(choice)==a.shape[run_axis], "underspecified choice"
    final_pos = run_axis - (1 if choice_axis<run_axis else 0)
    val = np.moveaxis(a, source=[run_axis, choice_axis], destination=[0,1])
    il = list(val.shape)
    il.pop(1)
    r = np.ndarray(shape=tuple(il), dtype=a.dtype)
    for i in range(len(choice)):
        r[i] = val[i,choice[i]]
    return np.moveaxis(r, source=0, destination=final_pos)
