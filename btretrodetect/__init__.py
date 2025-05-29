import numpy as np
import os
from glob import glob
import pickle
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import LinearSVC
import simplejpeg
import datetime    
import json
    
debugtimesum = 0
debugtime = datetime.datetime.now()
def debug_time_record(msg=None):
    global debugtime
    global debugtimesum    
    if msg is not None:
        elapsed = (datetime.datetime.now() - debugtime).total_seconds()
        debugtimesum+=elapsed
        print("%9.2f ms, %s" % (1000*elapsed,msg))
    else:
        if debugtimesum>0: print("%9.2f ms, TOTAL" % (1000*debugtimesum))
        debugtimesum=0
        
    debugtime = datetime.datetime.now()
    
    
kernelstore = {}

configpath = os.path.expanduser('~')+'/.btretrodetect/'
os.makedirs(configpath,exist_ok=True)

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
    stats.extend(getringstats(patch['img_patch']))    
    return stats

class TrainRetrodetectModel():
    def __init__(self,pathtodata,groupby='all',stopearly=None):
        """
        Manages a dictionary of photo_queues; each photo_queue can then be used to train
        the ML classifier separately. This class is just needed when training the models.
        
        The retrodetect algorithm is applied to each, and human labels searched for and associated.
        
        Parameters:
        - pathtodata: path to the data (this can be e.g. a session or set, etc.
        - groupby: the photo_queues generated could be amalgamated into either separate photo_queues by:
           - 'set'
           - 'camera' (so the same camera in different sets),
           - 'all' (all the data from all the cameras into a single photo_queue). [default]
        
        !! At the moment getting the camera or set from the live data is difficult, so we default to 'all'.
        Stores in 'photo_queues' parameter a dictionary of photo_queues, split depending on groupby.    
        """

        photo_queues = {}
        for root, dirs, files in os.walk(pathtodata):
            if 'btviewer' in root:
                pathtoimages = os.path.dirname(root)
                photo_queue = self.apply_retrodetect_and_associate_label(pathtoimages,stopearly)
                
                if groupby == 'set':
                    queue_name = '/'.join(os.path.normpath(pathtoimages).split(os.sep)[-3:])
                if groupby == 'camera':
                    queue_name = os.path.normpath(pathtoimages).split(os.sep)[-1]
                if groupby == 'all':
                    queue_name = 'all'                
                if queue_name not in photo_queues: photo_queues[queue_name] = []
                photo_queues[queue_name].extend(photo_queue)       
        self.photo_queues = photo_queues
        self.train_all_clfs()
        
    def build_patch_dataset(self,photo_queue,threshold_dim_patches=5):
        """
        Builds the paired input,output numpy arrays (X,y) for training using the data in photo_queue.
        
        Typically this would be called on a photo_queue generated by "self.apply_retrodetect_and_associate_label",
        as the images in photo_queue need to have been entered into Retrodetect.process_image sequentially,
        to add the imgpatches and need to have had the human-annotations added.

        Parameters:
          - photo_queue = a list of photoitems. Some of them should have a 
            "imgpatches" - a list of candidate patches in the image.
            "labeldata" - a list of human-annotated labels of where the tags really are in the image.
          - threshold_dim_patches = only include patches with a maximum value above this threshold, default 5.
        Returns
          - X = a 2d numpy array, each row is one patch, each column is a feature (from the summary stats)
          - y = boolean labels = whether this is a tag
        """
        patches = []
        labels = []
        
        no_labels_found_warning = True
        no_patches_found_warning = True
        for photoitem in photo_queue:    
            if 'labeldata' not in photoitem: continue
            no_labels_found_warning = False
            if 'imgpatches' not in photoitem: continue
            no_patches_found_warning = False
            for patch in photoitem['imgpatches']:       
                labelcoords = np.array([[label['x'], label['y']] for label in photoitem['labeldata']])
                patchcoord = np.array([patch['x'],patch['y']])            
                true_tag = np.min(np.linalg.norm(labelcoords-patchcoord,axis=1))<6              
                labels.append(true_tag)
                patches.append(patch)
        if no_labels_found_warning: print("No annotations found")
        if no_patches_found_warning: print("No patches found")    
        X = []
        y = []
        for patch,label in zip(patches,labels):
            try:
                stats = getstats(patch)
                if stats is None: continue #failed to generate stats
            except: #some of the patches are on the edge of the image, and we can't compute the stats for these
                continue
            if stats[1]<threshold_dim_patches: continue

            X.append(stats)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        #y = np.array(labels)

        return X,y
    
    def apply_retrodetect_and_associate_label(self,pathtoimages,stopearly=None):
        """
        Loads all the image files in this path, adds them to photo_queue,
        then applies the Retrodetect algorithm to them sequentially (as if
        it was running on the tracking box).
        """
        
        photo_queue = [] 
        for fn in sorted(glob(pathtoimages+'/*.np')):
            try:
                photoitem = np.load(fn,allow_pickle=True)
            except:# UnpicklingError:
                print("Failed to unpickle '%s'." % fn)
                continue
            photoitem['filename'] = fn
            photo_queue.append(photoitem)
            if stopearly is not None: 
                if len(photo_queue)>=stopearly: break
        print("Found %d files" % len(photo_queue))
        
        retrod = Retrodetect()
        for photoitem in photo_queue:
            retrod.process_image(photoitem)

        #for each photo, try to find the btviewer generated labelled location of the tag
        for photoitem in photo_queue:
            _,filename = os.path.split(photoitem['filename'])
            jsonfilename = os.path.join(pathtoimages,'btviewer',filename[:-2]+'json')
            import json
            try:
                labeldata = json.load(open(jsonfilename,'rb'))
                if len(labeldata)==0:
                    continue #skip           
                photoitem['labeldata'] = labeldata
            except FileNotFoundError:
                continue #skip
            
        return photo_queue

    
    def train_clf(self,X,y):
        """
        Build a classifier using the training data in X, y.
        """
        #clf = RandomForestClassifier(n_estimators = 100, class_weight={False:0.9, True:.1})  
        #clf = LogisticRegression()#class_weight={False:0.9, True:.1})
        clf = MLPClassifier()#class_weight={False:0.9, True:.1})
        #clf = LinearSVC(dual='auto')#,class_weight={False:0.99, True:.01})
        return clf.fit(X, y)
       
    def train_all_clfs(self):
        clfsfile = configpath+'clfs.pkl'
        try:
            clfs = pickle.load(open(clfsfile,'rb'))
        except:
            print("No classifiers found, making new file")
            clfs = {}
        oldkeynum = np.random.randint(1000000) #to be nice, I just move the old classifier to another key...
        for key, photo_queue in self.photo_queues.items():
            X,y = self.build_patch_dataset(photo_queue,8)
            if key in clfs:
                oldkey = key+'__%d' % oldkeynum
                print("Overwriting classifier for %s (old classifier moved to key='%s'" % (key,oldkey))
                clfs[oldkey] = clfs[key]
            print("Saving classifier as: %s" % key)
            clfs[key] = self.train_clf(X,y)
        
        print("Saving classifiers to %s" % clfsfile)
        print("List of classifiers currently saved:")
        for key in clfs.keys():
            print(key)
        pickle.dump(clfs,open(clfsfile,'wb'))


        
        

class Retrodetect:
    def __init__(self,Ndilated_keep = 20,Ndilated_skip = 5,Npatches = 20,patchSize = 16,patchThreshold=2,normalisation_blur=50,scalingfactor=5,base_path='/home/pi/beephotos'):
        self.scalingfactor = scalingfactor
        self.base_path = base_path
        self.Ndilated_keep = Ndilated_keep
        self.Ndilated_skip = Ndilated_skip
        self.Npatches = Npatches
        self.patchSize = patchSize
        self.delSize = patchSize #we'll just delete the same size for now?
        self.normalisation_blur = normalisation_blur
        self.Ndilated_use = Ndilated_keep - Ndilated_skip
        self.previous_dilated_imgs = None #keep track of previous imgs...
        self.patchThreshold = 4
        self.idx = 0
        self.imgcount = 0
        self.associated_colour_retrodetect = None
    
    def classify_patches(self,photoitem,groupby='camera'):
        clfsfile = configpath+'clfs.pkl'
        try:
            clfs = pickle.load(open(clfsfile,'rb'))
        except:
            print("No classifiers found.")
            return
            
        ##TODO I've added code to bee_track to now save this info into the photoitem,
        # photo_object['session_name'] = session_name
        # photo_object['set_name'] = set_name
        # photo_object['dev_id'] = self.devid.value
        # photo_object['camid'] = camid
        #but this data isn't in the examples I'm using. Later we should incorporate.
        
        #print("====")
        #print(photoitem['filename'])
        #if groupby=='camera':
        #    clfname = os.path.normpath(photoitem['filename']).split(os.sep)[-2]
        #if groupby=='set':
        #    clfname = '/'.join(os.path.normpath(photoitem['filename']).split(os.sep)[-4:-1])
        #if groupby=='all':
        clfname = 'all'

        if clfname not in clfs:
            print("Classifier not found for %s" % clfname)
            return None
        X = []
        for patch in photoitem['imgpatches']:
            stats = getstats(patch)
            if stats is None:
                res = False
            else:
                res = clfs[clfname].predict_proba(np.array(stats)[None,:])[0,1]
            patch['retrodetect_predictions'] = res
    
    def process_image(self,photoitem,groupby='camera'): ##TODO: PASS THIS METHOD THE CLASSIFIER WE WANT TO USE... AS IT WON'T HAVE ACCESS TO A FILENAME/PATH NECESSARILY
        tempdebugtime = datetime.datetime.now()
        if 'imgpatches' in photoitem:
            print('[already processed]')
            return
        if photoitem['img'] is None:
            print("no 'img' image object. can't process.")
            return
        debug_time_record() 
        print(photoitem['img'].dtype)
        
        smallmaxedimage = getblockmaxedimage(photoitem['img'],self.scalingfactor,1,resize=False)
        #smallmaxedimage = photoitem['img']
        debug_time_record('reduce image size')            
        raw_img = smallmaxedimage.astype(float)
        debug_time_record('convert to float')
        
        
        blurred = fast_gaussian_filter(raw_img,self.normalisation_blur)    
        debug_time_record('blur')
        img = raw_img/(1+blurred)
        debug_time_record('normalise with blurred image')  
        #photoitem['normalised_img'] = img.copy()
        blocksize = 1#6
        offset = 0#2
        dilated_img = img #getblockmaxedimage(img,blocksize,offset,resize=False)
        debug_time_record('dilate image computed')
        if self.previous_dilated_imgs is None:
            self.previous_dilated_imgs = np.zeros(list(dilated_img.shape)+[self.Ndilated_keep])
            #self.previous_imgs = np.zeros(list(photoitem['img'].shape)+[self.Ndilated_keep])
            debug_time_record('creating previous dilated images array (one off)')
        self.previous_dilated_imgs[:,:,self.idx] = dilated_img
        #self.previous_imgs[:,:,self.idx] = photoitem['img']
        
        debug_time_record('placing dilated image in previous dilated imgs array')
        self.idx = (self.idx + 1) % self.Ndilated_keep

        subtraction_img = np.max(self.previous_dilated_imgs[:,:,self.idx:(self.idx+self.Ndilated_use)],2)    
        debug_time_record('compute subtraction image')
        if self.idx+self.Ndilated_use>self.Ndilated_keep:
            other_subtraction_img = np.max(self.previous_dilated_imgs[:,:,:(self.idx-self.Ndilated_skip)],2)
            subtraction_img = np.max(np.array([other_subtraction_img,subtraction_img]),0)
            debug_time_record('alternative subtraction image calculation')
    
        self.imgcount+=1

        resized_subtraction_img = np.empty_like(img)
        debug_time_record('create empty array (size: '+str(resized_subtraction_img.shape)+')')
        insideimg = subtraction_img.repeat(blocksize,axis=0).repeat(blocksize,axis=1)
        debug_time_record('repeat blocks to rebuild subtraction image at larger size (size: '+str(insideimg.shape)+')')
        #resized_subtraction_img[:insideimg.shape[0],:insideimg.shape[1]] = insideimg    
        resized_subtraction_img[blocksize*offset:(blocksize*offset+insideimg.shape[0]),blocksize*offset:(blocksize*offset+insideimg.shape[1])] = insideimg
        debug_time_record('place insideimg inside the resized subtraction block')
        diff = img - resized_subtraction_img
        debug_time_record('compute diff')

        #photoitem['resized_subtraction_img'] = resized_subtraction_img.copy()
        

        #We need to temporarily keep this 'diff' image, as this is a handy way of finding the
        #a tag in the colour image near the one found in the greyscale image, but it needs to
        #be removed later.
        #photoitem['diff'] = diff.copy()
        print("Image shape")
        print(img.shape)
        if self.imgcount>2: #self.Ndilated_keep: #we have collected enough to start tracking...
            photoitem['imgpatches'] = []
            for p in range(self.Npatches):
                y,x = np.unravel_index(diff.argmax(), diff.shape)
                #photoitem['img']
                print(x,y)
                
                diff_max = diff[y,x]
                if diff_max<self.patchThreshold: break #we don't need to save patches that are incredibly faint.
                img_max = img[y,x]
                raw_max = raw_img[y,x]
                
                
                raw_patch = photoitem['img'][y*self.scalingfactor-self.patchSize:y*self.scalingfactor+self.patchSize,x*self.scalingfactor-self.patchSize:x*self.scalingfactor+self.patchSize].copy().astype(np.float32) 
                if self.scalingfactor>1:
                    try:
                        hires_blurred = rescalePatch(blurred,x*self.scalingfactor,y*self.scalingfactor,self.patchSize,self.scalingfactor)
                        img_patch = raw_patch/hires_blurred
                        diff_patch = img_patch - rescalePatch(resized_subtraction_img,x*self.scalingfactor,y*self.scalingfactor,self.patchSize,self.scalingfactor)
                    except Exception as e:
                        print("Skipping patch")
                        print(e)
                        #continue...?
                        continue #skip this patch?
                else:
                    img_patch = img[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].astype(np.float32).copy()
                    diff_patch = diff[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].copy().astype(np.float32)
                #raw_patch = raw_img[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].copy().astype(np.float32)     
                
                
                diff[max(0,y-self.delSize):min(diff.shape[0],y+self.delSize),max(0,x-self.delSize):min(diff.shape[1],x+self.delSize)]=-5000
                photoitem['imgpatches'].append({'raw_patch':raw_patch, 'img_patch':img_patch, 'diff_patch':diff_patch, 'x':x*self.scalingfactor, 'y':y*self.scalingfactor, 'diff_max':diff_max, 'img_max':img_max, 'raw_max':raw_max})
            
            #self.classify_patches(photoitem,groupby)
            debug_time_record('looping over %d patches...' % self.Npatches)
        photoitem['greyscale'] = True   
        photoitem['blurred'] = blurred         
        if self.associated_colour_retrodetect is not None:
            self.associated_colour_retrodetect.newgreyscaleimage(photoitem)
        debug_time_record('colour photo processing')
        self.save_image(photoitem,lowres_image=smallmaxedimage)
        debug_time_record('saving image')
        debug_time_record()
        print('TOTAL TIME TAG FINDING [greyscale]: ',(datetime.datetime.now() - tempdebugtime).total_seconds())


    def save_image(self,photoitem,fn=None,keepimg=None,lowres_image=None):
        """
        Save a (potentially compacted) version of the photoitem.
        - photoitem: the photoitem dictionary to save
        - fn: the filename to save to (if none, then it tries to save to base_path/datetime/session_name_compact/set_name_compact/dev_id/camid/compact_originalfilename)
            (where originalfilename comes from the photoitem['filename'])
        - keepimg: whether to keep the full image in the file. If photoitem['index'] is a multiple of 100 it keeps it, if this isn't set.
        - lowres_image: rather than encode the full size image into the jpeg, we can encode the lowres one. This also allows us to brighten it (a necessary step I've found
                            to avoid the jpeg algorithm just blanking the whole image). Although this looks terrible, so skipping for now!
        """    
        if fn is None:
            parents = "%s/%s/%s/%s/%s/%s/" % (self.base_path,datetime.date.today(),photoitem['session_name']+'_compact',photoitem['set_name']+'_compact',photoitem['dev_id'],photoitem['camid'])
            fn = parents + 'compact_' + photoitem['filename']
        print("===================")
        print(fn)
        print("===================")        
        compact_photoitem = photoitem #.copy() #saves time and memory if we actually keep [and trash] the photoitem!
        if keepimg is None: keepimg = photoitem['index']%100==0 #every 100 we keep the image
        
        if lowres_image is not None:
            scaledimg = lowres_image
            scalingfactor = self.scalingfactor
        else:
            scaledimg = photoitem['img']
            scalingfactor = 1
        scaledimg = scaledimg.astype(float)*10
        scaledimg[scaledimg>255]=255
        compact_photoitem['jpgimg'] = simplejpeg.encode_jpeg(scaledimg[:,:,None].astype(np.uint8),colorspace='GRAY',quality=40)
        compact_photoitem['jpgimg_processing'] = {'scalingfactor':scalingfactor, 'multiplier':10}
        if not keepimg: compact_photoitem['img'] = None
        
        #pickle.dump(compact_photoitem, open(fn,'wb'))
        
        try:
            pickle.dump(compact_photoitem, open(fn,'wb'))
        except FileNotFoundError:
            print("Parent Directory not found")
            os.makedirs(os.path.split(fn)[0])
            pickle.dump(compact_photoitem, open(fn,'wb'))
        
        
            
class ColourRetrodetect(Retrodetect):
    def __init__(self,Nbg_keep = 20,Nbg_skip = 5,normalisation_blur=50,patchSize=16,camid='all',base_path='/home/pi/beephotos',scalingfactor=5):
        self.scalingfactor=scalingfactor
        self.base_path = base_path
        offset_configfile = configpath+'offset.json'
        try:
            with open(offset_configfile,'r') as f:
                offsetdata = json.load(f)
                try:
                    self.offset = offsetdata[camid]
                except KeyError:

                    print('Offset config file does not include the specific camera key (%s). To set correctly, create a file %s containing a dictionary, e.g. {"%s": [20, 10]}.' % (camid,offset_configfile,camid))
                    raise Exception("No offset data available: Can't generate colour tag file!!!")                    
            print("Using %s offset file (%d, %d)" % (offset_configfile, self.offset[0], self.offset[1]))
        except FileNotFoundError:
            print('No offset file found!!! To set correctly, create a file %s containing a dictionary: {"%s": [20, 10]}.' % (offset_configfile,camid))
            raise Exception("No offset data available: Can't generate colour tag file!!!") 
        
            #self.offset = [0,0]
        assert len(self.offset)==2
        self.Nbg_keep = Nbg_keep
        self.Nbg_skip = Nbg_skip
        self.patchSize = patchSize
        self.normalisation_blur = normalisation_blur
        self.Nbg_use = Nbg_keep - Nbg_skip
        #self.previous_bg_imgs = None #keep track of previous imgs...
        self.idx = 0
        #self.imgcount = 0
        self.processed_photoitems = []

        
        self.unassociated_photoitems = []
        self.greyscale_photoitems = []
        
    def newgreyscaleimage(self,greyscale_photoitem):
        self.greyscale_photoitems.append(greyscale_photoitem)
        self.match_images()
        if len(self.greyscale_photoitems)>10:
            del self.greyscale_photoitems[0]
        
    def process_colour_image(self,photoitem,imgpatches):
        tempdebugtime = datetime.datetime.now()
        raw_img = photoitem['img'].astype(float)
        photoitem['imgpatches'] = []
        for patch in imgpatches:
            y,x = patch['y'],patch['x']
            x = x + self.offset[0]
            y = y + self.offset[1]
            x = 2*(x//2)
            y = 2*(y//2)
            #img_patch = img[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].astype(np.float32).copy()
            #diff_patch = diff[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].copy().astype(np.float32)        
            raw_patch = raw_img[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].copy().astype(np.float32)
            if 'retrodetect_predictions' in patch:
                pred = patch['retrodetect_predictions']
            else:
                pred = None
            photoitem['imgpatches'].append({'raw_patch':raw_patch, 'img_patch':None, 'diff_patch':None, 'x':x, 'y':y, 'retrodetect_predictions':pred})
        self.save_image(photoitem)
        print('TOTAL TIME [colour image]',(datetime.datetime.now() - tempdebugtime).total_seconds())
        
    def match_images(self):
        """
        Tries to match the images in self.unassociated_photoitems with those in self.greyscale_photoitems,
        this method is called after either a call to process_image (i.e. with a colour image added), or a call to
        newgreyscaleimage, with a new greyscale image.
        """
        for photo in self.unassociated_photoitems:
            match = [gs_photo for gs_photo in self.greyscale_photoitems if photo['record']['triggertime']==gs_photo['record']['triggertime']]
            if len(match)>0:
                print("MATCH: %s=%s" % (photo['filename'], match[0]['filename']))
                self.greyscale_photoitems.remove(match[0])
                photo['greyscale'] = False
                if ('imgpatches' not in match[0]): #this greyscale image doesn't have any patches
                    print("No patches")
                    match[0]['imgpatches'] = []
                    #self.unassociated_photoitems.remove(photo)
                    #continue
                photo['imgpatches'] = []
                
                    
                self.process_colour_image(photo,match[0]['imgpatches'])
                self.processed_photoitems.append(photo)
                if len(self.processed_photoitems)>10:
                    del self.processed_photoitems[0]
                #SAVE PHOTO!
                #super().save_image(photo)
                #photo['asssociated_gs_photoitem'] = match[0]
                #self.process_colour_image(photo,match[0])
                self.unassociated_photoitems.remove(photo)
                
        
    def process_image(self,photoitem):
        if photoitem is None:
            print("[photo none]")
            return
        if 'imgpatches' in photoitem:
            print('[already processed]')
            #return 
        #look for any matching photos...
        self.unassociated_photoitems.append(photoitem)
        self.match_images()
        

        if len(self.unassociated_photoitems)>10:
            del self.unassociated_photoitems[0]
