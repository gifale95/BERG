import os
import cortex
import nibabel
import numpy as np
import matplotlib
import glob
from mapdata.nsd_mapdata import nsd_mapdata
from mapdata.nsd_datalocation import nsd_datalocation


def list_files(dir_path):
    fileNames = []
    for f in os.listdir(dir_path):
        if os.path.isfile(dir_path+f):
            fileNames += [f,]
    return sorted(fileNames)


def convert_file(input_path, output_path, file_name, subject, sourcespace = 'func1pt8', targetspace = 'anat0pt8', interpmethod = 'nearest', force=False):
    ###
    if '.nii.gz' in file_name:
        l = 7
    elif '.nii' in file_name:
        l = 4
    else:
        print ('unrecognized file type')
        return
    
    sourcedata = f'{input_path}{file_name}'
    outputfile =f'{output_path}{file_name[:-l]}_{interpmethod}.nii'
    if os.path.isfile(outputfile) and not force:
        print (f'{outputfile} already exists')
    else:   
        targetdata = nsd_mapdata(
                subject,
                sourcespace,
                targetspace,
                sourcedata,
                interptype=interpmethod,
                badval=np.nan,
                outputfile=outputfile)
        print (f'{outputfile} has been converted')

def convert_dir(input_path, output_path, subject, sourcespace = 'func1pt8', targetspace = 'anat0pt8', interpmethod = 'nearest', force=False):
    for d in [f for f in list_files(input_path) if '.nii' in f]: 
        convert_file(input_path, output_path, d, subject, sourcespace = sourcespace,
            targetspace = targetspace, interpmethod = interpmethod, force=force)

        
def make_volume(data_file, data_file2=None, subject='subj01', transform='identity', \
            cmap='jet', vmin=0, vmax=1, vmin2=0, vmax2=1):
    data1 = nibabel.load(data_file)
    arr1 = data1.get_fdata().T.astype(float)
    if transform=='identity':   #identity isn't quite true, data that aligns to the subj anatomical must first be tipped forward 90deg then flipped left/right
        arr1 = np.flip((np.rot90(arr1, k=3)),2)
    if data_file2 is not None:
        data2 = nibabel.load(data_file2)
        arr2 = data2.get_fdata().T.astype(float)
        if transform=='identity':   #identity isn't quite true, data that aligns to the subj anatomical must first be tipped forward 90deg then flipped left/right
            arr2 = np.flip((np.rot90(arr2, k=3)),2)        
        return cortex.Volume2D(arr1, arr2, subject, transform, cmap=cmap, 
                   vmin=vmin, vmax=vmax, vmin2=vmin2, vmax2=vmax2)    
    else:
        return cortex.Volume(arr1, subject, transform, cmap=cmap, 
                   vmin=vmin, vmax=vmax)    

    
def webview(data_file, data_file2=None, subject='subj01', transform='identity', \
            cmap='jet', vmin=0, vmax=1, vmin2=0, vmax2=1):
    vol = make_volume(data_file, data_file2=data_file2, subject=subject, transform=transform, \
            cmap=cmap, vmin=vmin, vmax=vmax, vmin2=vmin2, vmax2=vmax2)
    return cortex.webgl.show(vol)


def make_flatmap(data_file, data_file2=None, subject='subj01', transform='identity', cmap='jet', \
                      vmin=0, vmax=1, vmin2=0, vmax2=1, thick=1,\
                      with_colorbar=False, surf_type='sulci', \
                      save_dir=None, dpi=600, bgcolor=None, 
                      show_image=True, with_rois=False):
    import matplotlib.pyplot as plt
    vol = make_volume(data_file, data_file2=data_file2, subject=subject, transform=transform, \
            cmap=cmap, vmin=vmin, vmax=vmax, vmin2=vmin2, vmax2=vmax2)
    # set parameters for selected texture/color of the cortical surface
    curv_contrast = None if surf_type == 'sulci' else 0
    curv_bright = None if surf_type == 'sulci' else 0 if surf_type == 'black' \
        else 1 if surf_type == 'white' else .5
    # make the figure; there are many more arguments that can be tweaked than what is shown here
    fig = cortex.quickflat.make_figure(vol, 
                                  recache=False, 
                                  pixelwise=True,
                                  thick=thick,
                                  sampler='nearest', 
                                  height=2048,
                                  depth=.5,
                                  with_rois=with_rois, 
                                  with_labels=False, 
                                  with_colorbar=with_colorbar,  
                                  with_dropout=False, 
                                  with_curvature=True,  
                                  curvature_brightness=curv_bright, 
                                  curvature_contrast=curv_contrast, 
                                  curvature_threshold=None, 
                                  colorbar_location=(.39, .95, .2, .04))
    # showing inline
    if show_image:
        plt.show() 
    # saving 
    if save_dir is not None:
        base_name = data_file.split('/')[-1][:-4]
        filename = "%s_flatmap_%s_%s_%.2fto%.2f" %(base_name, transform, cmap, vmin, vmax)
        filename = save_dir + filename.replace('.', 'pt') + '.png'
        print (filename)

        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
        if bgcolor is None:
            fig.savefig(filename, transparent=True, dpi=dpi)
        else:
            fig.savefig(filename, facecolor=bgcolor, transparent=False, dpi=dpi)
    fig.clf()
    plt.close()
