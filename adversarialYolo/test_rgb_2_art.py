from PIL import Image
from torchvision import transforms

def read_image(path, dim=3):
    """
    Read an input image to be used as a patch

    :param path: Path to the image to be read.
    :return: Returns the transformed patch as a pytorch Tensor.
    """
    patch_size = 300
    patch_img = Image.open(path).convert('RGB')
    tf = transforms.Resize((patch_size, patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()

    adv_patch_cpu = tf(patch_img)
    if(dim ==3):
        return adv_patch_cpu ##  torch.Size([3, 300, 300])
    else:
        return adv_patch_cpu[0].unsqueeze(0) ##  torch.Size([1, 300, 300])

def show(tensor_image):
    img = transforms.ToPILImage()(tensor_image.detach().cpu())
    img.show()

img = read_image("sample/fox.jpg")
# print("img size: "+str(img.size()))  ##  torch.Size([3, 300, 300])
# print("img     : "+str(img))

# resize
img = transforms.ToPILImage()(img.detach().cpu())
tf_s = transforms.Resize((30, 30))
tf_l = transforms.Resize((300, 300))
img_resize = tf_s(img)
img_resize = tf_l(img_resize)
img_resize.show()
