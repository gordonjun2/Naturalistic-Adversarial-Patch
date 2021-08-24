import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torch_tools.visualization import to_image
from visualization import interpolate
from loading import load_from_dir

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/SN_MNIST/',
#     G_weights='./models/pretrained/generators/SN_MNIST/')

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/SN_Anime/',
#     G_weights='./models/pretrained/generators/SN_Anime/')

deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/BigGAN/',
    G_weights='./models/pretrained/generators/BigGAN/G_ema.pth')

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/ProgGAN/',
#     G_weights='./models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth')

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/StyleGAN2/',
#     G_weights='./models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt')

discovered_annotation = ''
for d in deformator.annotation.items():
    discovered_annotation += '{}: {}\n'.format(d[0], d[1])
print('human-annotated directions:\n' + discovered_annotation)

"""--------------------------------------------------------------"""

from utils import is_conditional

rows = 8
plt.figure(figsize=(5, rows), dpi=250)

# set desired class for conditional GAN
# print(G)
if is_conditional(G):
    G.set_classes(12)

annotated = list(deformator.annotation.values())
print(annotated)
inspection_dim = annotated[1]
zs = torch.randn([rows, G.dim_z] if type(G.dim_z) == int else [rows] + G.dim_z, device='cuda')


for z, i in zip(zs, range(rows)):
    interpolation_deformed = interpolate(
        G, z.unsqueeze(0),
        shifts_r=16,
        shifts_count=4,
        dim=inspection_dim,
        deformator=deformator,
        with_central_border=True)

    plt.subplot(rows, 1, i + 1)
    plt.axis('off')
    grid = make_grid(interpolation_deformed, nrow=11, padding=1, pad_value=0.0)
    grid = torch.clamp(grid, -1, 1)

    plt.imshow(to_image(grid))
plt.show()