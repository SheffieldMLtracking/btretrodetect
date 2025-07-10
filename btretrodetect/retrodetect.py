import numpy as np
import os
from glob import glob
import pickle
import simplejpeg
import datetime    
import json
from btretrodetect.utils import fast_gaussian_filter, getblockmaxedimage, rescalePatch

from btretrodetect.utils import debug_time_record #temporary for profiling on box

from btretrodetect.utils import getstats

configpath = os.path.expanduser('~')+'/.btretrodetect/'
os.makedirs(configpath,exist_ok=True)
        

class Retrodetect:
    def __init__(self,Ndilated_keep = 20,Ndilated_skip = 5,Npatches = 40,patchSize = 16,patchThreshold=2,normalisation_blur=50,scalingfactor=2,base_path='/home/pi/beephotos',message_queue=None,skip_saving=False):
        self.scalingfactor = scalingfactor
        #print("Scaling factor set to %d" % self.scalingfactor)
        self.base_path = base_path
        self.Ndilated_keep = Ndilated_keep
        self.Ndilated_skip = Ndilated_skip
        self.Npatches = Npatches
        self.patchSize = patchSize
        self.delSize = patchSize #we'll just delete the same size for now?
        self.normalisation_blur = normalisation_blur
        self.Ndilated_use = Ndilated_keep - Ndilated_skip
        self.previous_dilated_imgs = None #keep track of previous imgs...
        self.message_queue = message_queue
        self.patchThreshold = patchThreshold #previously 4.
        self.idx = 0
        self.imgcount = 0
        self.associated_colour_retrodetect = None
        self.failed_to_classify_quiet = False
        self.skip_saving = skip_saving
        clfsfile = configpath+'clfs.pkl'
        try:
            self.clfs = pickle.load(open(clfsfile,'rb'))
            clfname = 'all'
            if clfname not in self.clfs:
                print("Classifier not found for %s" % clfname)
                self.clfs = None
            print("Using classifier %s in %s" % (clfname,clfsfile))
            #Temporary if-statement code, to handle change in the file structure...
            if 'classifier' in self.clfs[clfname]:
                self.clfs = self.clfs[clfname]['classifier']
            else:
                self.clfs = self.clfs[clfname]
        except:
            print("No classifiers found (looked in %s)." % clfsfile)
            self.clfs = None
        
    def classify_patches(self,photoitem,groupby='camera'):
        X = []
        for patch in photoitem['imgpatches']:
            if self.clfs is None: 
                res = None
            else:
                stats = getstats(patch)
                if stats is None:
                    res = False
                else:
                    res = self.clfs.predict_proba(np.array(stats)[None,:])[0,1]
            patch['retrodetect_predictions'] = res
    
    def process_image(self,photoitem,groupby='camera'): ##TODO: PASS THIS METHOD THE CLASSIFIER WE WANT TO USE... AS IT WON'T HAVE ACCESS TO A FILENAME/PATH NECESSARILY
        tempdebugtime = datetime.datetime.now()
        if photoitem is None:
            print("[photo none]")
            return
        if 'imgpatches' in photoitem:
            print('o',end="",flush=True)
            try:
                self.classify_patches(photoitem,groupby)
            except Exception as e:
                if not self.failed_to_classify_quiet:
                    print("\n==================================================\nFailed to classify (has the classifier been updated?):")
                    print(e)
                    print("Future failure messages muted\n==================================================\n")
                    self.failed_to_classify_quiet = True
                return
            
            return
        if photoitem['img'] is None:
            print("no 'img' image object. can't process (%s)." % photoitem['filename'])
            return
        debug_time_record() 

        if self.scalingfactor>1:
            print("Scaling image for processing: [scaling factor=%d] [threshold=%d]" % (self.scalingfactor,self.patchThreshold))
            smallmaxedimage = getblockmaxedimage(photoitem['img'],self.scalingfactor,1,resize=False)
        else:
            print("Image not scaled for processing: [scaling factor=%d] [threshold=%d]" % (self.scalingfactor,self.patchThreshold))
            smallmaxedimage = photoitem['img']
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
        if self.imgcount>2: #self.Ndilated_keep: #we have collected enough to start tracking...
            photoitem['imgpatches'] = []
            for p in range(self.Npatches):
                y,x = np.unravel_index(diff.argmax(), diff.shape)
                
                diff_max = diff[y,x]
                if diff_max<self.patchThreshold: break #we don't need to save patches that are incredibly faint.
                img_max = img[y,x]
                raw_max = raw_img[y,x]
                
                
                raw_patch = photoitem['img'][y*self.scalingfactor-self.patchSize:y*self.scalingfactor+self.patchSize,x*self.scalingfactor-self.patchSize:x*self.scalingfactor+self.patchSize].copy().astype(np.float32) 
                if self.scalingfactor>1:
                    try:
                        hires_blurred = rescalePatch(blurred,x*self.scalingfactor,y*self.scalingfactor,self.patchSize,self.scalingfactor)
                        img_patch = raw_patch/(hires_blurred+1)
                        diff_patch = img_patch - rescalePatch(resized_subtraction_img,x*self.scalingfactor,y*self.scalingfactor,self.patchSize,self.scalingfactor)
                    except Exception as e:
                        print("Skipping patch")
                        print(e)
                        #still need to delete the tag...
                        diff[max(0,y-self.delSize):min(diff.shape[0],y+self.delSize),max(0,x-self.delSize):min(diff.shape[1],x+self.delSize)]=-5000
                        continue #skip this patch?
                else:
                    img_patch = img[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].astype(np.float32).copy()
                    diff_patch = diff[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].copy().astype(np.float32)
                #raw_patch = raw_img[y-self.patchSize:y+self.patchSize,x-self.patchSize:x+self.patchSize].copy().astype(np.float32)     
                
                
                diff[max(0,y-self.delSize):min(diff.shape[0],y+self.delSize),max(0,x-self.delSize):min(diff.shape[1],x+self.delSize)]=-5000
                photoitem['imgpatches'].append({'raw_patch':raw_patch, 'img_patch':img_patch, 'diff_patch':diff_patch, 'x':x*self.scalingfactor, 'y':y*self.scalingfactor, 'diff_max':diff_max, 'img_max':img_max, 'raw_max':raw_max})
            
            self.classify_patches(photoitem,groupby)
            debug_time_record('looping over %d patches...' % self.Npatches)
        photoitem['greyscale'] = True   
        #photoitem['blurred'] = blurred         
        if self.associated_colour_retrodetect is not None:
            self.associated_colour_retrodetect.newgreyscaleimage(photoitem)
        debug_time_record('colour photo processing')
        self.save_image(photoitem,lowres_image=smallmaxedimage)
        debug_time_record('saving image')
        debug_time_record()
        #print('TOTAL TIME TAG FINDING [greyscale]: ',(datetime.datetime.now() - tempdebugtime).total_seconds())


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
        
        if self.skip_saving: return
        if fn is None:
            parents = "%s/%s/%s/%s/%s/%s/" % (self.base_path,datetime.date.today(),photoitem['session_name']+'_compact',photoitem['set_name']+'_compact',photoitem['dev_id'],photoitem['camid'])
            fn = parents + 'compact_' + photoitem['filename']
        #print(fn)
        #print("===================")        
        compact_photoitem = photoitem #.copy() #saves time and memory if we actually keep [and trash] the photoitem!
        if keepimg is None: keepimg = photoitem['index']%300==0 #every 300 we keep the image
        
        if lowres_image is not None:
            scaledimg = lowres_image
            scalingfactor = self.scalingfactor            
            #quality = 40
        else:
            scaledimg = photoitem['img']#[::5,::5] #quick hack to get filesize down of colour images...
            scalingfactor = 1
            #quality = 40
            
        morescaling = 1
        if scalingfactor<=3:
            morescaling = 6//scalingfactor #10/2 = 5
            scaledimg = scaledimg[::morescaling,::morescaling]            
            scalingfactor = scalingfactor * morescaling
        quality = 20
                    
        scaledimg = scaledimg.astype(float)*10
        scaledimg[scaledimg>255]=255
        
        compact_photoitem['jpgimg'] = simplejpeg.encode_jpeg(scaledimg[:,:,None].astype(np.uint8),colorspace='GRAY',quality=quality)
        compact_photoitem['jpgimg_processing'] = {'scalingfactor':scalingfactor, 'multiplier':10}
        if not keepimg: compact_photoitem['img'] = None
        
        #pickle.dump(compact_photoitem, open(fn,'wb'))
        
        try:
            pickle.dump(compact_photoitem, open(fn,'wb'))
            print("Saved compact photoitem: %s" % fn.split('/')[-1])
            if self.message_queue is not None: self.message_queue.put("Saved compact photoitem: %s" % fn.split('/')[-1])
        except FileNotFoundError:
            print("Parent Directory not found")
            os.makedirs(os.path.split(fn)[0])
            pickle.dump(compact_photoitem, open(fn,'wb'))
        

