import numpy as np
import os
from glob import glob
import pickle
import datetime    
import json
from btretrodetect.retrodetect import Retrodetect

configpath = os.path.expanduser('~')+'/.btretrodetect/'
os.makedirs(configpath,exist_ok=True)

class ColourRetrodetect(Retrodetect):
    def __init__(self,Nbg_keep = 20,Nbg_skip = 5,normalisation_blur=50,patchSize=16,camid='all',base_path='/home/pi/beephotos',scalingfactor=5,message_queue=None,skip_saving=False):
        self.message_queue=message_queue
        self.scalingfactor=scalingfactor
        self.base_path = base_path
        self.skip_saving = skip_saving
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
        #print('TOTAL TIME [colour image]',(datetime.datetime.now() - tempdebugtime).total_seconds())
        
    def match_images(self):
        """
        Tries to match the images in self.unassociated_photoitems with those in self.greyscale_photoitems,
        this method is called after either a call to process_image (i.e. with a colour image added), or a call to
        newgreyscaleimage, with a new greyscale image.
        """
        for photo in self.unassociated_photoitems:
            match = [gs_photo for gs_photo in self.greyscale_photoitems if photo['record']['triggertime']==gs_photo['record']['triggertime']]
            if len(match)>0:
                #print("MATCH: %s=%s" % (photo['filename'], match[0]['filename']))
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
