import numpy as np


def gaussian_mask(n, sigma=None):
    if sigma is None:
        sigma = 0.3 * (n // 2) + 0.8
    X = np.arange(-(n//2), n//2+1)
    kernel = np.exp(-(X**2)/(2*sigma**2))
    return kernel


def seperable_conv(I, filter_x, filter_y):
    h, w = I.shape[:2]
    n = filter_x.shape[0]//2
    I_a = np.zeros(I.shape)
    I_b = np.zeros(I.shape)
    for x in range(n, w-n):
        patch = I[:, x-n:x+n+1]
        I_a[:, x] = np.sum(patch * filter_x, 1)
    filter_y = np.expand_dims(filter_y, 1)
    for y in range(n, h-n):
        patch = I_a[y-n:y+n+1, :]
        I_b[y, :] = np.sum(patch * filter_y, 0)
    return I_b


def detect(I, n_g=5, n_w=5, k=0.06):
    h, w = I.shape
    sobel_1 = np.array([-1, 0, 1])
    sobel_2 = np.array([1, 2, 1])
    I_x = seperable_conv(I, sobel_1, sobel_2)
    I_y = seperable_conv(I, sobel_2, sobel_1)
    g_kernel = gaussian_mask(n_g)
    I_x = seperable_conv(I_x, g_kernel, g_kernel)
    I_y = seperable_conv(I_y, g_kernel, g_kernel)
    D_temp = np.zeros((h, w, 2, 2))
    D_temp[:, :, 0, 0] = np.square(I_x)
    D_temp[:, :, 0, 1] = I_x * I_y
    D_temp[:, :, 1, 0] = D_temp[:, :, 0, 1]
    D_temp[:, :, 1, 1] = np.square(I_y)
    g_filter = gaussian_mask(n_w)
    g_filter = np.dstack([g_filter] * 4).reshape(n_w, 2, 2)
    D = seperable_conv(D_temp, g_filter, g_filter)
    P = D[:, :, 0, 0]
    Q = D[:, :, 0, 1]
    R = D[:, :, 1, 1]
    T1 = (P + R) / 2
    T2 = np.sqrt(np.square(P - R) + 4 * np.square(Q)) / 2
    L_1 = T1 - T2
    L_2 = T1 + T2
    C = L_1 * L_2 - k * np.square(L_1 + L_2)
    return C, I_x, I_y, L_1, L_2
