import numpy as np
import cv2
import h5py


def load_s3_mat(s3_name, NORM=False):
    print('--Loading s3 file "{}"'.format(s3_name))
    f = h5py.File(s3_name,'r')
    s3 = f.get('RGB_S3')
    s3 = np.array(s3)
    s3 = s3.astype(np.float32)
    if NORM:
        print('--Normalizing...')
        s3 = (s3-np.min(s3))/(np.max(s3)-np.min(s3))
    s3 = np.moveaxis(s3, 0, -1) 
    
    return s3

def load_s3gt_mat(s3gt_name, NORM=False):
    print('--Loading s3 file "{}"'.format(s3gt_name))
    f = h5py.File(s3gt_name,'r')
    s3 = f.get('S3_SIM')
    s3 = np.array(s3)
    s3 = s3.astype(np.float32)
    if NORM:
        print('--Normalizing...')
        s3 = (s3-np.min(s3))/(np.max(s3)-np.min(s3))
    s3 = np.moveaxis(s3, 0, -1) 
    return s3

def load_s2_mat(s2_name, RESIZE_SHAPE=None, NORM=False):
    print('--Loading s2 file "{}"'.format(s2_name))
    f = h5py.File(s2_name,'r')
    s2 = f.get('RGB_S2')
    s2 = np.array(s2)
    s2 = s2.astype(np.float32)
    s2 = np.moveaxis(s2, 0, -1) 
    if NORM:
        print('--Product pre-processing...')
        s2 = np.clip(s2.astype(np.float32)/10000, 0, 1) # recovering the original reflectances of s2 (see https://forum.step.esa.int/t/dn-to-reflectance/15763/8, https://gis.stackexchange.com/questions/233874/what-is-the-range-of-values-of-sentinel-2-level-2a-images)
        print('--Normalizing...')
        s2 = (s2-np.min(s2))/(np.max(s2)-np.min(s2))
    if RESIZE_SHAPE: # reducing s2 to s3 size?
        print('--Resizing...')
        s2 = cv2.resize(s2, RESIZE_SHAPE, cv2.INTER_LANCZOS4).astype(np.float32) 
        if s2.ndim<3: # to recover the spectral dimension if necessary (cv2.resize removes the 3rd dimsion if it is 1) 
            s2 = np.expand_dims(s2, -1)
    return s2


def extract_patches(s3, s2, s3_gt=None, PATCH_SIZE=32, STEP=32, MAX_PATCH_SIZE=64):
    
    RATIO = s2.shape[0]//s3.shape[0]
    PATCH_s3 = PATCH_SIZE
    PATCH_s2 = PATCH_SIZE*RATIO
    STEP_s3 = STEP
    STEP_s2 = STEP*RATIO
    MAX_PATCH_s3 = MAX_PATCH_SIZE
    MAX_PATCH_s2 = MAX_PATCH_SIZE*RATIO

    list_s3_patches = []
    list_s2_patches = []
    list_s3_gt_patches = []

    print('--Extracting s3 patches...')
    for r in range(MAX_PATCH_s3//2,s3.shape[0]-MAX_PATCH_s3//2+1,STEP_s3):
        for c in range(MAX_PATCH_s3//2,s3.shape[1]-MAX_PATCH_s3//2+1,STEP_s3):
            ini_row = r-PATCH_s3//2
            end_row = r+PATCH_s3//2
            ini_col = c-PATCH_s3//2
            end_col = c+PATCH_s3//2
            list_s3_patches.append(s3[ini_row:end_row,ini_col:end_col])
    print('--Stacking patches...')

    s3 = np.stack(list_s3_patches, axis=0) # (num_patches, patch_s3, patch_s3, 21)
    
    print('--Extracting s2 patches...')
    for r in range(MAX_PATCH_s2//2,s2.shape[0]-MAX_PATCH_s2//2+1,STEP_s2):
        for c in range(MAX_PATCH_s2//2,s2.shape[1]-MAX_PATCH_s2//2+1,STEP_s2):
            ini_row = r-PATCH_s2//2
            end_row = r+PATCH_s2//2
            ini_col = c-PATCH_s2//2
            end_col = c+PATCH_s2//2
            list_s2_patches.append(s2[ini_row:end_row,ini_col:end_col])
    print('--Stacking patches...')
    s2 = np.stack(list_s2_patches, axis=0) # (num_patches, patch_s2, patch_s2, 12)

    if s3_gt is not None:
        print('--Extracting s3_gt patches...')
        for r in range(MAX_PATCH_s3//2,s3_gt.shape[0]-MAX_PATCH_s3//2+1,STEP_s3):
            for c in range(MAX_PATCH_s3//2,s3_gt.shape[1]-MAX_PATCH_s3//2+1,STEP_s3):
                ini_row = r-PATCH_s3//2
                end_row = r+PATCH_s3//2
                ini_col = c-PATCH_s3//2
                end_col = c+PATCH_s3//2
                list_s3_gt_patches.append(s3_gt[ini_row:end_row,ini_col:end_col])            
        print('--Stacking patches...')
        s3_gt = np.stack(list_s3_gt_patches, axis=0) # (num_patches, patch_s3, patch_s3, 21)

    return [s3, s2, s3_gt]


def load_data_products(s3_names, s2_names, s3_gt_names, patch_size, step, max_patch_size, downscale_s2=True, use_single_channel=True, NORM=True):

    # we expect a lists as inputs
    if not(isinstance(s3_names, list)): s3_names = [s3_names]
    if not(isinstance(s2_names, list)): s2_names = [s2_names]
    if not(isinstance(s3_gt_names, list)): s3_gt_names = [s3_gt_names]

    if not(len(s3_names)==len(s2_names) and len(s3_names)==len(s3_gt_names)):
        raise Exception("Different number of S3/S2/S3_gt products!")

    all_s3, all_s2, all_s3_gt = [], [], []
    all_s3_shapes, all_s2_shapes = [], []
    
    for i in range(len(s3_names)):

        print('--Loading {}...'.format(s3_names[i]))
        s3 = load_s3_mat(s3_names[i], NORM)
        print('--Loading {}...'.format(s2_names[i]))
        if downscale_s2:
            s2 = load_s2_mat(s2_names[i], (s3.shape[0],s3.shape[1]), NORM)
        else:
            s2 = load_s2_mat(s2_names[i], None , NORM)
        print('--Loading {}...'.format(s3_gt_names[i]))
        s3_gt = load_s3gt_mat(s3_gt_names[i], NORM)
        if use_single_channel:
            print('--Selecting a single channel for registration...')
            s3 = np.expand_dims(s3[:,:,1],axis=-1)#16 
            s2 = np.expand_dims(s2[:,:,1],axis=-1)#8 
            s3_gt = np.expand_dims(s3_gt[:,:,1],axis=-1)#16 

        print('--Extracting coupled patches...')
        x,y,z = extract_patches(s3, s2, s3_gt, patch_size, step, max_patch_size)

        # channels first for pytorch (num_patches, patch_size, patch_size, bands) --> (num_patches, bands, patch_size, patch_size)
        if x.ndim ==4: x = np.moveaxis(x, -1, 1) 
        if y.ndim ==4: y = np.moveaxis(y, -1, 1)
        if z.ndim ==4: z = np.moveaxis(z, -1, 1)

        all_s3.append(x)
        all_s2.append(y)
        all_s3_gt.append(z)
        all_s3_shapes.append(s3.shape)
        all_s2_shapes.append(s2.shape)

    print('-Stacking all patches together...')
    all_s3 = np.concatenate(all_s3, axis=0) # (num_patches, bands, patch_size, patch_size)
    all_s2 = np.concatenate(all_s2, axis=0)
    all_s3_gt = np.concatenate(all_s3_gt, axis=0)

    return [all_s3, all_s2, all_s3_gt, all_s3_shapes, all_s2_shapes]


def build_image_from_patches(patches, img_size, STEP=32, MAX_PATCH_SIZE=64):
    
    # patches --> (num_patches, bands, patch_size, patch_size)
    # img_size --> (out_img_rows, out_img_columns)
    PATCH_SIZE = patches.shape[-1]
    
    R, C, B = img_size[0], img_size[1], patches.shape[1]
    img = np.zeros([R, C, B]).astype(patches.dtype)
    
    i=0
    for r in range(MAX_PATCH_SIZE//2,img_size[0]-MAX_PATCH_SIZE//2+1,STEP):
        for c in range(MAX_PATCH_SIZE//2,img_size[1]-MAX_PATCH_SIZE//2+1,STEP):
    
            ini_row = r-PATCH_SIZE//2
            end_row = r+PATCH_SIZE//2
            ini_col = c-PATCH_SIZE//2
            end_col = c+PATCH_SIZE//2
            
            img[ini_row:end_row,ini_col:end_col,:] = np.moveaxis(patches[i,:,:,:], 0, -1)
            i += 1
            
    return img


from skimage.exposure import rescale_intensity
    
# Adjusting the dynamic range of a S2 image likewise in SNAP visualization (https://gis.stackexchange.com/questions/259907/constrast-stretching-sentinel-2-l1c)
def imadjust(img): 
    p1, p2 = np.percentile(img, (1, 99))
    out = rescale_intensity(img, in_range=(p1, p2), out_range=(0,1))
    return out   




 
