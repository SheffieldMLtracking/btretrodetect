import numpy as np
import os
from glob import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import LinearSVC
import simplejpeg
import datetime    
import json
import re
    
configpath = os.path.expanduser('~')+'/.btretrodetect/'
os.makedirs(configpath,exist_ok=True)

#######################DEBUG METHODS###############################
debugtimesum = 0
debugtime = datetime.datetime.now()
def debug_time_record(msg=None):
    global debugtime
    global debugtimesum    
    if msg is not None:
        elapsed = (datetime.datetime.now() - debugtime).total_seconds()
        debugtimesum+=elapsed
        #print("%9.2f ms, %s" % (1000*elapsed,msg))
    else:
        #if debugtimesum>0: print("%9.2f ms, TOTAL" % (1000*debugtimesum))
        debugtimesum=0
        
    debugtime = datetime.datetime.now()
    


##################TAG FINDING / BACKGROUND SUBTRACTION METHODS############################
#cache of kernels for gaussian filter    
kernelstore = {}
def gaussian_filter(img, sigma,croptosize=True):
    """
    Applies a gaussian smoothing kernel on a 2d img
    using numpy's 1d convolve function.
    uses 'reflect' mode for handling borders.

    img: the 2d numpy array
    sigma: the standard deviation (one value) of the Gaussian
    croptosize: whether to crop back to the shape of the img
                     (default True).

    Returns an image of the smae shape, unless
    croptosize is set to False, in which case the
    full added reflection is included.
    """
    L = sigma
    
    global kernelstore
    if sigma not in kernelstore:
        kernel = np.exp(-np.arange(-L*4,L*4)**2/(2*sigma**2)) #this is so fast, caching might be a bit pointless...
        kernel/= np.sum(kernel)
        kernelstore[sigma] = kernel
    else:
        kernel = kernelstore[sigma]
        
    inputimg = np.zeros(np.array(img.shape)+L*4)
    inputimg[L*2:-L*2,L*2:-L*2] = img
    inputimg[0:L*2,0:L*2] = img[L*2:0:-1,L*2:0:-1]
    inputimg[0:L*2,-L*2:-1] = img[L*2:0:-1,-L*2:-1]
    inputimg[-L*2:-1,0:L*2] = img[-L*2:-1,L*2:0:-1]
    inputimg[-L*2:-1,-L*2:-1] = img[-L*2:-1,-L*2:-1]
   

#    inputimg[0:L*2,0:L*2] = img[L*2:0:-1,L*2:0:-1]
#    inputimg[0:L*2,0:L*2] = img[L*2:0:-1,L*2:0:-1]            
    
    inputimg[0:L*2,L*2:-L*2] = img[L*2:0:-1,:]
   
    inputimg[-L*2:-1,L*2:-L*2] = img[:-L*2:-1,:]
    inputimg[L*2:-L*2,0:L*2] = img[:,L*2:0:-1]
    inputimg[L*2:-L*2,-L*2:-1] = img[:,:-L*2:-1]
    blurredimg = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, inputimg)
    blurredimg = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, blurredimg)
    if croptosize: blurredimg = blurredimg[L*2:-L*2,L*2:-L*2]
    return blurredimg

def fast_gaussian_filter(img, sigma, blocksize = 6):
    """
    Applies a gaussian smoothing kernel on a 2d img
    using numpy's 1d convolve function.
    uses 'reflect' mode for handling borders.

    (runs over 100 times faster than
     from scipy.ndimage import gaussian_filter
     and seems to do a reasonable approximation!)

    img: the 2d numpy array
    sigma: the standard deviation (one value) of the Gaussian
    blocksize: the size of the squares used.

    Returns an image of the smae shape as img.
    
    """
    k = int(img.shape[0] / blocksize)
    l = int(img.shape[1] / blocksize)
    smallimg = img[:k*blocksize,:l*blocksize].reshape(k,blocksize,l,blocksize).mean(axis=(-1,-3)) #from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
    smallimg[1:-1,:] = (smallimg[1:-1,:]+smallimg[2:,:]+smallimg[:-2,:])/3
    smallimg[:,1:-1] = (smallimg[:,1:-1]+smallimg[:,2:]+smallimg[:,:-2])/3    
    blurredimg = gaussian_filter(smallimg, int(sigma/blocksize),croptosize=False)    
    out_img = np.empty_like(img)
    insideimg = blurredimg.repeat(blocksize,axis=0).repeat(blocksize,axis=1)

    out_img = insideimg[sigma*2:sigma*2+out_img.shape[0],sigma*2:sigma*2+out_img.shape[1]]
    return out_img
    
def getblockmaxedimage(img,blocksize, offset,resize=True):
    """
    Effectively replaces each pixel with approximately the maximum of all the
    pixels within offset*blocksize of the pixel (in a square).
    
    Get a new image of the same size (if resize=True), but filtered
    such that each square patch of blocksize has its maximum calculated,
    then a search box of size (1+offset*2)*blocksize centred on each pixel
    is applied which finds the maximum of these patches.
    
    img = image to apply the filter to
    blocksize = size of the squares
    offset = how far from the pixel to look for maximum
    """

    k = int(img.shape[0] / blocksize)
    l = int(img.shape[1] / blocksize)
    if blocksize==1:
        maxes = img
    else:
        maxes = img[:k*blocksize,:l*blocksize].reshape(k,blocksize,l,blocksize).max(axis=(-1,-3)) #from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
    templist = []
    
    if offset>1:
        xm,ym = maxes.shape
        i = 0
        for xoff in range(-offset+1,offset,1): #(if offset=1, for xoff in [0]) (if offset=2, for xoff in [-1,0,1])...
          for yoff in range(-offset+1,offset,1):
            if i==0:
              max_img = maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset]
            else:
              max_img = np.maximum(max_img,maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
            i+=1
    else:
        max_img = maxes

    if resize:
        out_img = np.full_like(img,0)
        inner_img = max_img.repeat(blocksize,axis=0).repeat(blocksize,axis=1)
        out_img[blocksize*offset:(blocksize*offset+inner_img.shape[0]),blocksize*offset:(blocksize*offset+inner_img.shape[1])] = inner_img
    else:
        out_img = max_img
    return out_img    
    
def rescalePatch(im,x,y,patchSize,blocksize):
    """
    Returns a patchSize*2 x patchSize*2 array of a square around point x,y, from image in im;
    the x and y coordinates relate to a scaled-up image, one that is 'blocksize' times larger than im.

    no interpolation.
    """
    if (x<patchSize) or (y<patchSize) or ((x+patchSize+blocksize)//blocksize>im.shape[1]) or ((y+patchSize+blocksize)//blocksize>im.shape[0]):
        raise ValueError("part of patch outside image. Patch coordinate: (x=%d,y=%d), Image shape: (shape[0]=%d,shape[1]=%d), blocksize=%d, patchSize=%d" % (x,y,im.shape[0],im.shape[1],blocksize,patchSize))
    a = im[(y-patchSize)//blocksize:(y+patchSize+blocksize)//blocksize,(x-patchSize)//blocksize:(x+patchSize+blocksize)//blocksize]
    a = a.repeat(blocksize,axis=0).repeat(blocksize,axis=1)
    a = a[(y-patchSize)%blocksize:(patchSize*2)+(y-patchSize)%blocksize,(x-patchSize)%blocksize:(patchSize*2)+(x-patchSize)%blocksize]
    return a
    
#############################METHODS TO SUPPORT TRAINING#####################
def getringstats(patchimg):
    """
    Generates summary statistics based on the values of pixels in concentric rings around the centre of a patch of a tag.
    Parameters:
     patchimg: a 32x32 patch image
    Returns:
     list of 9 values:
       the minimum, mean and maximum of the pixels at either 1.5, 3 or 5 pixel radii from the centre
    """
    assert patchimg.shape[0]==32
    assert patchimg.shape[1]==32    
    stats = []    
    for radius in [1.5,3,5]:
        coords = [[int(radius*np.cos(angle)), int(radius*np.sin(angle))] for angle in np.linspace(0,2*np.pi,1+int(radius*2),endpoint=False)]
        coords = np.array(coords)+16
        vals = patchimg[coords[:,0],coords[:,1]]
        minval, meanval, maxval = np.min(vals), np.mean(vals), np.max(vals)
        stats.extend([minval, meanval, maxval])
    return stats
    
def getstats(patch):
    """
    Generates summary statistics based on the values of pixels in a patch.
    Parameters:
     patch: a dictionary of patch info including raw_max, img_max and diff_max, each should be a 32x32 patch
    Returns:
     list of 12 values:
       - the maximum value of the initial raw patch image, the maximum value of the normalised image patch, and the maximum value of the difference image
       (once previous images have been subtracted).
       - the minimum, mean and maximum of the pixels at either 1.5, 3 or 5 pixel radii from the centre
    """
    if patch['img_patch'].shape!=(32,32):
        return None
        
    stats = [patch['raw_max'],patch['img_max'],patch['diff_max']]
    stats.extend(getringstats(patch['diff_patch']))    
    return stats
    
#########################METHODS TO HELP WORK WITH FILES############################
def isgreyscale(photoitem):
    """
    Returns true if the photo item is from a greyscale camera
    TODO Need to add code to bee_track to record in the photoitem if it's greyscale. Currently a hack to try to guess from the image itself.   
    
    """
    print("Trying to determine if %s is greyscale..." % photoitem['filename'])
#    print(photoitem)
    if 'greyscale' in photoitem: 
        gs = photoitem['greyscale']
        print("Determined to be greyscale=%s from 'greyscale' photoitem feature" % ("True" if gs else "False"))
        return gs   
    gs = np.mean(np.abs(photoitem['img'][::2,::2].astype(float)-photoitem['img'][1::2,1::2].astype(float)))/np.mean(photoitem['img'][::2,::2].astype(float))<1
    
#    print(np.mean(np.abs(photoitem['img'][1:-1:2,:].astype(float)-photoitem['img'][0:-2:2,:]/2-photoitem['img'][2::2,:]/2))/np.mean(photoitem['img']))
#    print(np.mean(np.abs(photoitem['img'][1::2,1::2].astype(float)-photoitem['img'][1::2,::2])))
#    print(np.mean(np.abs(photoitem['img'][::2,1::2].astype(float)-photoitem['img'][1::2,::2])))    
    print("Determined to be greyscale=%s from photoitem img" % ("True" if gs else "False"))
    return gs   
    

def totalsecs(st):
    """
    Tries to get the number of seconds from the filename passed in 'st'
    """
    try:
        time_hms = [int(s) for s in re.findall('([0-9]{1,2})[:+]([0-9]{2})[:+]([0-9]{2})',st)[0]]
    except TypeError as e:
        print("Tried to find time of format HH:MM:SS in '%s'" % st)
        raise e
    return time_hms[0]*3600 + time_hms[1]*60 + time_hms[2]

def get_cam_paths(pathtocameras):
    """
    Returns a path to each of the cameras on this box, in a dictionary (of two items!),
    with a key of 0 --> not greyscale, 1 --> greyscale.
    
    Pass it the path to one of the cameras.
    """
    #splitpath = os.path.normpath(pathtoimages).split(os.sep)
    camerapaths = {}
    for possiblepath in glob(pathtocameras+'/*'):
        fns = sorted(glob(possiblepath+'/*.np'))
        if len(fns)==0: continue
        print("OPENING %s..." % fns[0])
        photoitem = pickle.load(open(fns[0],'rb'))
        greyscale = None
        if possiblepath.split('/')[-1][:2]=='M-':
            print("Determined to be greyscale from path (%s)" % possiblepath)
            greyscale = True
        if possiblepath.split('/')[-1][:2]=='C-':
            print("Determined to be colour from path (%s)" % possiblepath)
            greyscale = False
        if greyscale is None: greyscale = isgreyscale(photoitem)
        camerapaths[greyscale] = possiblepath
    return camerapaths
