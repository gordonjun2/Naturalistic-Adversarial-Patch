import torch
import numpy as np

a = torch.rand((1, 1, 5, 5))
print(a)

# x1 = 2.5, x2 = 4.5, y1 = 0.5, y2 = 3.5
# out_w = 2, out_h = 3
size = torch.Size((1, 1, 3, 2))
print(size)

# theta
theta_np = np.array([[0.5, 0, 0.75], [0, 0.75, 0]]).reshape(1, 2, 3)
theta = torch.from_numpy(theta_np)
print('theta:')
print(theta)
print()

flowfield = torch.nn.functional.affine_grid(theta, size)
sampled_a = torch.nn.functional.grid_sample(a, flowfield.to(torch.float32))
sampled_a = sampled_a.numpy().squeeze()
print('sampled_a:')
print(sampled_a)

# compute bilinear at (0.5, 2.5), using (0, 3), (0, 4), (1, 3), (1, 4)
# quickly compute(https://blog.csdn.net/lxlclzy1130/article/details/50922867)
print()
coeff = np.array([[0.5, 0.5]])
A = a[0, 0, 0:2, 2:2+2]
print('torch sampled at (0.5, 3.5): %.4f' % sampled_a[0,0])
print('numpy compute: %.4f' % np.dot(np.dot(coeff, A), coeff.T).squeeze())