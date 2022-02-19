"""
Dataset Library for ACDC dataset
"""
import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from torch.utils.data import Dataset



def read_h5py(path):
  """
  Function to read .h5 files.
  Arguements:
      path: Path to the .h5 file
  Output:
      Data in the file as numpy array
  """	
  with h5py.File(path, "r") as f:
    print(f"Reading from {path} ====================================================")
    print("Keys in the h5py file : %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data1 = np.array((f[a_group_key]))
    print(f"Number of samples : {len(data1)}")
    print(f"Shape of each data : {data1.shape}")
    return data1

def calculate_2dft(input):
    """
    Function to compute the 2D-Discrete Fourier Transform
    of the given image. 
    """
    ft = np.fft.fft2(input)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    """
    Function to compute the Inverse 2D-Discrete Fourier 
    Transform of the given fourier map. 
    """
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    return ift.real

def add_motion_artefacts(img):
  """
  Function to generate synthethic ACDC dataset
  by add motion artefacts to the image frames. 
  Arguments:
        img : Input image frames from ACDC dataset
              of a slice. The size of the image is 
              [H,W,S]. H -> Heigth, W-> Width,
              S -> Number of phases or frames
  Output:
        motion_artefact_added: Motion artefact added synthethic
                               image frames in the shape of 
                               [H,W,S']. Here S' = S-2*N where N = 7

  """
  N = 9
  start_ind = N
  end_ind = img.shape[2]-N
  motion_artefact_added = []
  gt =[]
  eps = 10e-11

  for i in range(start_ind,end_ind):
    fft_mixed = calculate_2dft(img[:,:,i])

    # K-space mixing
    for j in range(0,2*N+1): 
      fft_2d = calculate_2dft(img[:,:,(i+j)-N])
      fft_mixed[:,j*11:(j+1)*11] = fft_2d[:,j*11:(j+1)*11]
      # if j<N:
      #   fft_mixed[:,j*6:(j+1)*6] = fft_2d[:,j*6:(j+1)*6]
      # elif j == N:
      #   fft_mixed[:,j*6: 58] = fft_2d[:,j*6:58]
      # elif j>N:
      #   fft_mixed[:,58+(j-N-1)*6: 58+(j-N)*6] = fft_2d[:,58+(j-N-1)*6: 58+(j-N)*6]

    # Zero Padding
    fft_mixed[0:30,:] = 0
    fft_mixed[-30:,:] = 0
    fft_mixed[:,0:30] = 0
    fft_mixed[:,-30:] = 0

    ma_img = calculate_2dift(fft_mixed)
    motion_artefact_added.append(ma_img)
    gt.append(img[:,:,i])		
  motion_artefact_added = np.array(motion_artefact_added)
  gt = np.array(gt)
  return motion_artefact_added,gt



class CineCardiac(Dataset):
    """
    Dataset class for the synthethic ACDC dataset. Here the
    class loads the croped data in the size of (H,W)=(100,100) from the 
    original dataset then generated synthesized image by adding motion artefact
    to the image frames. 
    
    Arguements:
      data_list : list of paths to the crop images saved as .h5 files
      device    : cuda or cpu
      transform : transform to be added to the image frames (tranform.ToTensor()).
    
    Outputs:
      img     : Motion artefact added synthethic ACDC data with size of 
                [B,S,H,W] ==> B - Batch size,S = 7 -> Number of phases or frames,
                H -> Heigth, W-> Width,
      img_inv : img in the reverse order in terms of S
      gt      : Ground truth images         
        
    """
    def __init__(self, data_list, device,data_type = None,  transform=None, target_transform=None):
        
        first = True
        for base in data_list:
          if first:
            self.img_data = np.reshape(read_h5py(base),(1,100,100,30))
            first = False
          else:
            self.img_data = np.concatenate((self.img_data,np.reshape(read_h5py(base),(1,100,100,30))),axis = 0)

        print(self.img_data.shape)    
        self.device = device
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
      gt = (self.img_data[idx] - np.min(self.img_data[idx]))/(np.max(self.img_data[idx]) - np.min(self.img_data[idx])) #Normalizing to [0,1]
      img,gt = add_motion_artefacts(gt)
      img = np.moveaxis(img,0,-1)
      gt = np.moveaxis(gt,0,-1)
      # img = img[:,:,9-3:9+4]
      # img_inv = img_inv[:,:,9-3:9+4]
      # gt = gt[:,:,9-3:9+4]
      img = img[:,:,6-3:6+4]
      img_inv = np.zeros((100,100,7))
      for i in range(7):
          img_inv[:,:,i] = img[:,:,6-i]
      gt = gt[:,:,6-3:6+4]
      # print(img.shape,gt.shape)
      if self.transform:
          img = self.transform(img).to(self.device)  
          img_inv = self.transform(img_inv).to(self.device)  
          gt  = self.transform(gt).to(self.device)  
      return img, img_inv, gt