import struct
import numpy as np
import math

class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=np.float32):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
    
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=np.float32, **kwargs):
        super(linspace, self).__init__(n_div, dtype=np.float32, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = np.float32(width)/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = np.float32(width)/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=np.float32, **kwargs):
        super(logspace, self).__init__(n_div, dtype=np.float32, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)
    
def model_space(model_specs):
    vm = np.asarray(model_specs[0])
    nt = np.prod([sms.length for sms in model_specs[1]])           
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(model_specs[1])]
    xs, ys, ss = np.meshgrid(rx, ry, rs, indexing='ij')    
    return np.concatenate([xs.reshape((nt,1)).astype(dtype=np.float32), 
                           ys.reshape((nt,1)).astype(dtype=np.float32), 
                           ss.reshape((nt,1)).astype(dtype=np.float32)], axis=1)      

def model_space_pyramid(sigmas, min_spacing, aperture):
    rf = []
    for s in sigmas:
        X, Y = np.meshgrid(np.linspace(-aperture/2, aperture/2, int(np.ceil(aperture/(s * min_spacing)))),
                           np.linspace(-aperture/2, aperture/2, int(np.ceil(aperture/(s * min_spacing)))))
        rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    return np.concatenate(rf, axis=0)
