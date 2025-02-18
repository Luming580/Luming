import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
"""
 @Time    : 2024/07/05 19:06
 @Author  : ***
 @E-mail  : ***
 
 @Project : RDNet
 @File    : GFwithSobel.py
 @Function: GaborFilter
 
"""
def analysis_window_r0(q, len, number):
    device = q.device
    # Parameters
    n1 = len
    n2 = number
    L = n1
    N = n2
    N0 = n2
    M = int(L / N0)
    M0 = int(L / N)
    q = q.to(device)  # Ensure q is on the correct device

    # Initialize V vector (ensure it's complex)
    V = torch.zeros((M0 * N0, 1), dtype=torch.complex64, device=device)
    V[0] = L / (M * N)

    # Create matrix H using broadcasting
    k = torch.arange(L, dtype=torch.float32, device=device)
    m = torch.arange(M0, dtype=torch.float32, device=device)
    n = torch.arange(N0, dtype=torch.float32, device=device)

    k1 = (k[None, :] + m[:, None] * N) % L
    exp_term = torch.sqrt(torch.sqrt(torch.tensor(2.0, device=device)) / q) * torch.exp(
        -torch.pi * ((k1 - (len / 2)) / q) ** 2)

    exp_term = exp_term[:, None, :]

    cos_term = torch.cos(2 * torch.pi * n[:, None] * k / N0)
    sin_term = torch.sin(2 * torch.pi * n[:, None] * k / N0)

    cos_term = cos_term[None, :, :]
    sin_term = sin_term[None, :, :]

    H = exp_term * (cos_term + 1j * sin_term)
    H = H.reshape(M0 * N0, L)

    # Compute pseudo-inverse and r0
    H_conj = torch.conj(H)
    G = torch.linalg.pinv(torch.matmul(H, torch.conj(H)))
    G = G.to(torch.complex64)
    r0 = torch.matmul(torch.matmul(H_conj, G), V)

    # Create reference window h
    j = torch.arange(L, dtype=torch.float32, device=device)
    h = torch.sqrt(torch.sqrt(torch.tensor(2.0, device=device)) / q) * torch.exp(
        -torch.pi * ((j - len / 2) / q) ** 2)

    rr = torch.matmul(r0, r0.T)  # Signal power distribution
    hh = torch.outer(h, h)

    return rr, hh


def sobel_x():
    """Sobel x kernel"""
    kernel = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
    return kernel.view(1, 1, 3, 3)


def sobel_y():
    """Sobel y kernel"""
    kernel = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=torch.float32)
    return kernel.view(1, 1, 3, 3)


def calculate_gradients(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    sobel_kernel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)

    g_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1)
    g_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1)

    M = torch.abs(g_x) + torch.abs(g_y)

    Diff = M / image_tensor

    MN = image_tensor.shape[2] * image_tensor.shape[3]
    Q = torch.sum(torch.sum(Diff)) / (MN + 2)

    return Q


def adaptive_window_width(feature_block):
    device = feature_block.device
    kernel_size = feature_block.shape[1]

    sobel_kernel_x = sobel_x().to(device).repeat(kernel_size, 1, 1, 1)
    sobel_kernel_y = sobel_y().to(device).repeat(kernel_size, 1, 1, 1)

    grad_x = F.conv2d(feature_block, sobel_kernel_x, padding=1, groups=kernel_size)
    grad_y = F.conv2d(feature_block, sobel_kernel_y, padding=1, groups=kernel_size)

    # grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = torch.abs(grad_x) + torch.abs(grad_y)

    max_grad = torch.max(grad_magnitude.view(grad_magnitude.shape[0], -1), dim=1)[0].mean()
    m = max_grad / (kernel_size) ** 2
    q = torch.pow(1 / (20 * m), 0.25) * 5

    return q

# RGNet: Reconstruction-generation Network
class GaborFilter(nn.Module):
    def __init__(self, dim, size=32, num=4, len=128, number=16, num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        # self.complex_weight = nn.Parameter(
        #     torch.randn(1, num, num, size, size, dim, dtype=torch.float32) * 0.02)  # [512, 128, 128]

        assert dim % num_blocks == 0, f"hidden_size {dim} should be divisble by num_blocks {num_blocks}"

        self.len = len
        self.number = number
        self.dim = dim
        self.size = size
        self.num = num

        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.dim // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bias = x  # torch.Size([1, 512, 32, 32])
        B = x.shape[0]
        C = x.shape[1]
        a = x.shape[2]
        b = x.shape[3]
        agray = x.reshape(B, a, b, C)  # torch.Size([1, 32, 32, 512])
        q = adaptive_window_width(agray)

        rr, hh = analysis_window_r0(q, len=self.len, number=self.number)

        x = torch.zeros((B, self.num, self.num, self.size, self.size, C)).to(device)

        for m in range(self.num):  # 0->1
            for n in range(self.num):  # 0->1
                m_range = np.mod(np.arange(-m * self.size, a - m * self.size), a)
                n_range = np.mod(np.arange(-n * self.size, b - n * self.size), b)
                R2 = torch.tensor(rr[m_range][:, n_range]).to(device)
                R = agray * R2.unsqueeze(-1)

                R1 = torch.zeros(1, self.size, self.size, C).to(device)

                for i1 in range(self.num):  # 0->1
                    for i2 in range(self.num):  # 0->1
                        R1 = R1 + R[:, i1 * self.size:(i1 + 1) * self.size,
                                  i2 * self.size:(i2 + 1) * self.size, :]

                FT = torch.fft.fftn(R1, dim=(1, 2))
                x[:, m, n, :, :, :] = torch.real(FT) - torch.imag(FT)
        x = x.reshape(B, a, b, self.num_blocks, self.block_size)

        o1_real = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :], self.w1) + \
            self.b1
        )

        o2_real = (
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :], self.w2) + \
                self.b2
        )
        x = F.softshrink(o2_real, lambd=self.sparsity_threshold)
        x = x.reshape(B, self.num, self.num, self.size, self.size, C)

        aa = torch.zeros((B, self.num, self.num, self.size, self.size, C)).to(device)

        for m in range(self.num):  # 0->1
            for n in range(self.num):  # 0->1
                x1 = x[:, m, n, :, :, :].view(B, self.size, self.size, C)
                Rf = torch.fft.fft2(x1, dim=(1, 2))
                aa[:, m, n, :, :, :] = torch.real(Rf) - torch.imag(Rf)

        I = None
        for qx in range(self.num):
            dw = None
            for qy in range(self.num):
                II = torch.zeros(self.size, self.size).unsqueeze(0).unsqueeze(-1).to(device)
                for m in range(self.num):
                    for n in range(self.num):
                        gmn = torch.tensor(hh[torch.arange((qx - m) * self.size, (qx + 1 - m) * self.size) % a,
                                              torch.arange((qy - n) * self.size, (qy + 1 - n) * self.size) % b]).to(
                            device)
                        aa1 = aa[:, m, n, :, :, :].view(B, self.size, self.size, C)
                        II = II + aa1 * gmn.unsqueeze(0).unsqueeze(-1)

                if qy == 0:
                    dw = II
                else:
                    dw = torch.cat((dw, II), dim=2)

            if qx == 0:
                I = dw
            else:
                I = torch.cat((I, dw), dim=1)

        x = I.float()  # torch.Size([1, 64, 64, 512])
        x = x.reshape(B, x.shape[3], x.shape[1], x.shape[2])

        return x + bias



