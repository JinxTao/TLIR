import os
import argparse
import numpy as np
import pydicom
import os
import torch
from torch_radon import Radon, RadonFanbeam

def FP(x,n_angles=512,image_size=256):
    fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    radon_fanbeam = RadonFanbeam(image_size, fanbeam_angles, source_distance=256, det_distance=512, det_spacing=1.9,det_count=1024)
    sinogram = radon_fanbeam.forward(x)
    # mindata = torch.zeros((512,1024))
    # for i in range(170):
    #     mindata[i,:] = sinogram[i,:]
    # return sinogram.cpu().numpy(),mindata.cpu().numpy()
    return sinogram

def FBP(x,n_angles=512,image_size=256):
    fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    radon_fanbeam = RadonFanbeam(image_size, fanbeam_angles, source_distance=256, det_distance=512, det_spacing=1.9,det_count=1024)
    filtered_sinogram = radon_fanbeam.filter_sinogram(x)
    fbp = radon_fanbeam.backprojection(filtered_sinogram)
    return fbp

def replacement(input_image, sinogram):
    input_sino = FP(input_image)
    flag = 0
    for i in range(sinogram.shape[0]):
        if sinogram[i].sum()==0:
            flag = i
            break
    for i in range(flag):
        input_sino[i] = sinogram[i]
    return FBP(input_sino)

def conformance(input_image, sinogram, namda):
    device = input_image.device
    input_sino = FP(input_image)
    res_sino = input_sino - sinogram
    
    output_image = input_image - namda*FBP(res_sino)
    return output_image

def data_self_enhancement_module(input_image, sinogram, out_iter, namda, R=3):
    if out_iter%R == 0:
        output_image = replacement(input_image, sinogram)
    else:
        output_image = conformance(input_image, sinogram,namda)
    return output_image