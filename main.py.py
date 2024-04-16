
import numpy as np
import glob
import os
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch

from spectral import save_rgb
from scipy.io import savemat 
from codes.data import *
from codes.mycodes import get_gradient_3d, RegNet, U_Net2, Grad
from codes.losses import NCC
from codes.spatial_transformer import SpatialTransformer
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from PIL import Image

sys.path.append('/REGISTER/')

print('Loading dataset file names...')

s3_names = glob.glob('/REGISTER/data/S3/*.mat'); s3_names.sort()
s2_names = glob.glob('/REGISTER/data/S2/*.mat'); s2_names.sort()
s3_gt_names = glob.glob('/REGISTER/data/S3_GT/*.mat'); s3_gt_names.sort()

num_s3 = len(s3_names); 
num_s2 = len(s2_names); 
num_s3_gt = len(s3_gt_names); 
print('--Total s3 images:{}'.format(num_s3))
print('--Total s2 images:{}'.format(num_s2))
print('--Total s3_gt images:{}'.format(num_s3_gt))
if len(set([num_s3, num_s2, num_s3_gt]))>1:
    raise Exception("Different number of S3/S2/S3_gt products!")

patch_size = 352 # the patch needs to be 16*n because of the number of considered encoder layers
step = patch_size
max_patch_size = patch_size

GPU = "0" 

NUM_EPOCHS = 8000
BATCH_SIZE = 1
LERNING_RATE = 0.0001
int_downsize = 1
SINGLE = True
# selecting the files to process from ESA10 dataset
FILES = [0,1,2,3,4,5,6,7,8,9]

out_folder = '/REGISTER/results/'


print('Iterating over coupled S3/S2 images...')
for i in FILES:

    print('--Loading S3, S2, S3_GT')
    s3, s2, s3_gt, s3_shapes, s2_shapes = load_data_products(s3_names[i], s2_names[i], s3_gt_names[i], patch_size, step, max_patch_size, downscale_s2=False, use_single_channel=SINGLE, NORM=False)
    xtra, ytra, ztra = s3, s2, s3_gt
    s3_shape = s3_shapes[0][:-1] # shape for re-building (366,366)
    s2_shape = s2_shapes[0][:-1] # shape for re-building (366,366) or (5280,5280)
    RATIO = s2_shape[0]//s3_shape[0]
    
    if SINGLE == False:
        CHAN_S2 = s2.shape[1] 
        CHAN_S3 = s3.shape[1]
    else:
        CHAN_S2 = 1
        CHAN_S3 = 1

    print('--Model definition')
    img_shape = (patch_size,patch_size)
    regnet = RegNet(img_shape, RATIO, CHAN_S2, CHAN_S3)
    test =U_Net2(img_ch=2,output_ch=16)
    summary(test,(2,352,352),1,device="cpu")
    FILE_MODEL = os.path.join(out_folder, '%04d.pt' % NUM_EPOCHS)

    if not os.path.isfile(FILE_MODEL):

        print('--Model training')
        cudnn.benchmark = True
        device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')
        # prepare the model for training and send to device
        model = regnet.to(device)
        model.train()
        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LERNING_RATE)

        # prepare losses
        losses  = [NCC(RATIO).loss] + [Grad('l2', loss_mult=int_downsize).loss] 
        weights = [1] + [0.5]

        tensor_xtra = torch.Tensor(xtra.astype(np.float32)); # transform to torch tensors
        tensor_ytra = torch.Tensor(ytra.astype(np.float32))
        tradata = TensorDataset(tensor_xtra,tensor_ytra) # create datset
        traloader = DataLoader(dataset=tradata,batch_size=BATCH_SIZE, shuffle=True) # create dataloader

        # training loops
        for epoch in range(NUM_EPOCHS):

            for iteration, batch in enumerate(traloader):
                inputs, y_true = batch[0].to(device), batch[1].to(device)
                # run inputs through the model to produce a warped image and flow field
                y_pred  = model(inputs, y_true)                
                # calculate total loss
                loss = 0
                loss_list = []
                for n, loss_function in enumerate(losses):
                    curr_loss = loss_function(y_true, y_pred[n]) * weights[n]
                    loss_list.append('%.6f' % curr_loss.item())
                    loss += curr_loss
                loss_info = 'loss: %.6f  (%s)' % (loss.item(), ', '.join(loss_list))
                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # print step info
            epoch_info = 'Epoch: %04d' % (epoch + 1)
            print('  '.join((epoch_info, loss_info)), flush=True)

        # model save
        torch.save(model.state_dict(), os.path.join(out_folder, '%04d.pt' % NUM_EPOCHS))



    print('--Model testing')
    cudnn.benchmark = True
    device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')
    # prepare the model for training and send to device
    model = regnet.to(device)
    model.eval()
    
    state_dict = model.state_dict()
    for n, p in torch.load(FILE_MODEL, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    xtst = xtra
    ytst = ytra
    ztst = ztra
    
    tensor_xtst = torch.Tensor(xtst.astype(np.float32)); # transform to torch tensors
    tensor_ytst = torch.Tensor(ytst.astype(np.float32))
    tstdata = TensorDataset(tensor_xtst,tensor_ytst) # create datset
    tstloader = DataLoader(dataset=tstdata,batch_size=BATCH_SIZE, shuffle=False) # create sampler

    LIST_MOVED = []
    LIST_WARP = []

    with torch.no_grad():
                
        for iteration, batch in enumerate(tstloader):

            input_moving, input_fixed = batch[0].to(device), batch[1].to(device)

            moved, warp = model(input_moving, input_fixed, registration=True)
            
            moved= moved.cpu().detach().numpy().astype(np.float32)
            warp= warp.cpu().detach().numpy().astype(np.float32)

            LIST_MOVED.append(moved)
            LIST_WARP.append(warp)


    MOVED = np.concatenate(LIST_MOVED, axis=0) # (num_patches, 1, patch_s3, patch_s3)
    WARP = np.concatenate(LIST_WARP, axis=0) # (num_patches, 1, patch_s3, patch_s3, 12)
    del LIST_MOVED, LIST_WARP
    
    

    img_s3 = build_image_from_patches(xtst, s3_shape, step, max_patch_size) 
    img_s3_reg = build_image_from_patches(MOVED, s3_shape, step, max_patch_size)
    img_s2 = build_image_from_patches(ytst, s2_shape, step*RATIO, max_patch_size*RATIO)
    img_s3_gt = build_image_from_patches(ztst, s3_shape, step, max_patch_size)
    flow = WARP

    print('--Saving registered bands')

    save_rgb(os.path.join(out_folder, '{:02d}_s3.jpg'.format(i+1)), img_s3)
    save_rgb(os.path.join(out_folder, '{:02d}_s3_reg.jpg'.format(i+1)), img_s3_reg)
    save_rgb(os.path.join(out_folder, '{:02d}_s3_gt.jpg'.format(i+1)), img_s3_gt)
    save_rgb(os.path.join(out_folder, '{:02d}_s2.jpg'.format(i+1)), img_s2)

    if SINGLE == True:

        print('--Applying the uncovered transformation to all bands')
        
        s3, s2, s3_gt, s3_shapes, s2_shapes = load_data_products(s3_names[i], s2_names[i], s3_gt_names[i], patch_size, step, max_patch_size, downscale_s2=True, use_single_channel=False, NORM=False)
        xtst, ytst, ztst = s3, s2, s3_gt 
        s3_shape = s3_shapes[0][:-1] 
        s3_bands = s3_shapes[0][-1] 

        xtst_reg = np.zeros(s3_shapes[0]).astype(np.float32)

        tensor_xtst = torch.Tensor(xtst.astype(np.float32))
        pos_flow = torch.Tensor(warp.astype(np.float32))

        model.eval()

        color_out =np.zeros([patch_size,patch_size,3]).astype(np.float32)
        color = get_gradient_3d(patch_size,patch_size,(0,0,192),(255,255,64),(True,False,False))
        tensor_color = torch.Tensor(color.astype(np.float32))
        Image.fromarray(np.uint8(color)).save('/REGISTER/results/color.jpg',quality=95)
        transformer = SpatialTransformer([patch_size,patch_size])
       # pdb.set_trace()
        with torch.no_grad():
                    
            for b in range(s3_bands):
                
                ori_band = tensor_xtst[:,b].unsqueeze(1)
                reg_band = model.transformer(ori_band.to(device), pos_flow.to(device))
                xtst_reg[0:xtst.shape[2],0:xtst.shape[3],b] = reg_band.cpu().detach().numpy().astype(np.float32)   

            for d in range(3):
                
                orig_band = tensor_color[:,:,d].unsqueeze(0).unsqueeze(0)
                reg_band = model.transformer(orig_band.to(device), pos_flow.to(device))
                color_out[0:reg_band.shape[2],0:reg_band.shape[3],d] = reg_band.cpu().detach().numpy().astype(np.float32) 

        Image.fromarray(np.uint8(color_out)).save('/REGISTER/results/color.jpg',quality=95)
        mdic = {"I": color_out}
        savemat(os.path.join(out_folder, '{:02d}C.mat'.format(i+1)), mdic)  
        print('--Saving registered image as ".mat"')   
        mdic = {"I": xtst_reg}
        savemat(os.path.join(out_folder, '{:02d}I.mat'.format(i+1)), mdic)
        mdic = {"F": flow}
        savemat(os.path.join(out_folder, '{:02d}F.mat'.format(i+1)), mdic)
    else: 
        print('--Saving registered image as ".mat"')
        mdic = {"I": img_s3_reg}
        savemat(os.path.join(out_folder, '{:02d}.mat'.format(i+1)), mdic)
        mdic = {"F": flow}
        savemat(os.path.join(out_folder, '{:02d}.mat'.format(i+1)), mdic)
    
