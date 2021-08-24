import torch
import time
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
from PIL import Image
from GANLatentDiscovery.visualization import interpolate_shift
import glob
from pathlib import Path
import re


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def pad_and_scale(img, lab, imgsize):
    """

    Args:
        img:

    Returns:

    """
    if torch.is_tensor(img):
        if len(img) == 1 and len(img.size()) == 4:
            img = img[0]
        trans = transforms.ToPILImage(mode='RGB')
        if(img.is_cuda):
            img = img.cpu()
        img = trans(img)
    w,h = img.size
    if w==h:
        padded_img = img
    else:
        dim_to_pad = 1 if w<h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h,h), color=(127,127,127))
            padded_img.paste(img, (int(padding), 0))
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127,127,127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h  / w)
    resize = transforms.Resize((imgsize,imgsize))
    padded_img = resize(padded_img)     #choose here
    padded_img = transforms.ToTensor()(padded_img).unsqueeze(0)
    return padded_img, lab

def pad_lab(lab, max_n_labels):
    pad_size = max_n_labels - lab.shape[0]
    if(pad_size>0):
        padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
    else:
        padded_lab = lab
    return padded_lab

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def init_tensorboard(path=None ,name=None):
    if name is not None and path is not None:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        # time_str = time.strftime("%Y%m%d_38")
        # time_str = time.strftime("%Y%m%d_test")
        output_file_name = time_str + "_" + str(name)
        writer_dir = path / output_file_name
        print("init_tensorboard / time: "+str(writer_dir))
        # return SummaryWriter(f'runs/{output_file_name}')
        return SummaryWriter(writer_dir)
    else:
        print("init_tensorboard ("+str(name)+")")
        return SummaryWriter()

def denorm(img_tensors, stats=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def save_samples(index, patch, sample_dir, show=False):
    fake_images = patch
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()
def save_samples_simple(index, latent_tensors, generator, sample_dir, show=False):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()
def save_samples_GANLatentDiscovery(method_num,
                                    index, sample_dir, 
                                    deformator, G, 
                                    latent_shift, param_rowPatch_latent, fixed_rand_latent, 
                                    max_value_latent_item, 
                                    enable_shift_deformator, 
                                    device,
                                    mask_images_input=None):
    '''
    latent_shift: latent_shift_biggan
    param_rowPatch_latent: alpha, ex:0.99
    param_rowPatch_latent: rand
    '''
    if(method_num == 2):
        rows = 1
        zs = torch.randn([rows, G.dim_z] if type(G.dim_z) == int else [rows] + G.dim_z).to(device)
        z = zs[0]
        if(enable_shift_deformator):
            z = ((1-param_rowPatch_latent)*z) + (param_rowPatch_latent*fixed_rand_latent)
            latent_shift = torch.clamp((max_value_latent_item*latent_shift),max=max_value_latent_item,min=-max_value_latent_item).to(device)
            interpolation_deformed = interpolate_shift(G, z.unsqueeze(0),
                                    latent_shift=latent_shift,
                                    deformator=deformator,
                                    device=device)
        else:
            z = ((1-param_rowPatch_latent)*z) + (param_rowPatch_latent*latent_shift)
            interpolation_deformed = interpolate_shift(G, z.unsqueeze(0),
                                    latent_shift=torch.zeros_like(z).to(device),
                                    deformator=deformator,
                                    device=device)

        interpolation_deformed = interpolation_deformed.unsqueeze(0)   # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed + 1) * 0.5
        #
    elif(method_num == 3):
        rows = 1
        zs = torch.rand([rows, G.latent_size] if type(G.latent_size) == int else [rows] + G.latent_size, device=device)
        z = latent_shift
        interpolation_deformed = G(z.unsqueeze(0))
        # st()
        interpolation_deformed = interpolation_deformed.to(device)   # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed+1) * 0.5
        fake_images = torch.clamp(fake_images,0,1)
        # 
    elif(method_num == 4):
        fake_images = torch.zeros(1,3,640,640).to(device)
        for a_i, a in enumerate(mask_images_input):
            for b_i, b in enumerate(a):
                latent_shift_part = latent_shift[a_i,b_i*128:b_i*128+128]
                # print(latent_shift_part.size())
                fake_part = G(latent_shift_part)[0][0] * mask_images_input[a_i, b_i]
                # print(fake_part.size())
                fake_images[:,:,a_i*32:a_i*32+32,b_i*32:b_i*32+32] = fake_part
    
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)

def save_the_patched(index, the_patched, sample_dir, show=False):
    output_fname = 'patched-images-{0:0=4d}.png'.format(index)
    save_image(the_patched, os.path.join(sample_dir, output_fname), nrow=8)
    print('Saving', output_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(the_patched.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()