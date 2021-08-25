import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans

def split_color(img_tensor, n_colors):
    """
    [input]
    img_tensor : Analyze pictures of main colors. Default size(3,h,w)
    n_colors   : The amount of color left
    """
    n_items = len(img_tensor.size())
    if(n_items == 3):
        pixels = img_tensor.view(3,-1).float() ## size(3   , h*w)
        pixels = pixels.transpose(0,1)         ## size(h*w , 3  )
        # kmeans
        labels, palette = kmeans(
            X=pixels, num_clusters=n_colors, max_iter=200, tol=0.1, termination=0, distance='euclidean', device=torch.device('cuda:0'), verbose=0
        )
        _, counts = torch.unique(labels, return_counts=True)
        indices = torch.argsort(counts, descending=True)
        palette_sorted = palette[indices]
        counts_sorted = counts[indices]
        # print(palette_sorted)
        return palette_sorted, counts_sorted
    elif (n_items == 5):
        b,f,d,h,w = img_tensor.size()
        pixels = img_tensor.view(b,f,d,-1).float() ## size(b, f, 3, h*w)
        pixels = pixels.transpose(-1,-2)           ## size(b, f, h*w, 3)
        pixels = pixels.view(-1,h*w,d).float()     ## size(b*f , h*w, 3)
        bt_palette = 0
        bt_counts  = 0
        first_time = True
        for bt_p in pixels:
            # kmeans
            labels, palette = kmeans(
                X=bt_p, num_clusters=n_colors, max_iter=200, tol=0.1, termination=0, distance='euclidean', device=torch.device('cuda:0'), verbose=0
            )
            _, counts = torch.unique(labels, return_counts=True)
            indices = torch.argsort(counts, descending=True)
            palette_sorted = palette[indices]
            counts_sorted = counts[indices]
            if(first_time):
                first_time = False
                bt_palette = palette_sorted.unsqueeze(0)
                bt_counts  = counts_sorted.unsqueeze(0)
            else:
                bt_palette = torch.cat((palette_sorted.unsqueeze(0), bt_palette), 0)
                bt_counts  = torch.cat((counts_sorted.unsqueeze(0), bt_counts), 0)
            # print(palette)
        b_t_palette = bt_palette.view(b,f,bt_palette.size()[-2],bt_palette.size()[-1])
        b_t_counts  = bt_counts.view(b,f,-1)
        return b_t_palette, b_t_counts
