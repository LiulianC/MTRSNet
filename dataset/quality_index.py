try:
    from skimage.measure import compare_ssim, compare_psnr  
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    
from functools import partial
import numpy as np
from skimage.metrics import structural_similarity



def cal_bwpsnr(Y, X):
    batch_size, h, w, c = Y.shape
    psnr_values = []
    for i in range(batch_size):
        win_size = min(7, h, w)
        win_size = win_size if win_size % 2 == 1 else win_size - 1

        psnr_val = compare_psnr(
            Y[i], X[i], data_range=Y[i].max() - Y[i].min()
        )
        psnr_values.append(psnr_val)

    return np.array(psnr_values)

def cal_bwssim(Y, X):
    batch_size, h, w, c = Y.shape
    ssim_values = []
    for i in range(batch_size):
        win_size = min(7, h, w)
        win_size = win_size if win_size % 2 == 1 else win_size - 1
        
        ssim_val = structural_similarity(
            Y[i], X[i],
            data_range=1,
            win_size=win_size,
            channel_axis=-1 if c > 1 else None  
        )
        ssim_values.append(ssim_val)
    
    return np.array(ssim_values)

def compare_ncc(x, y):
    return np.mean((x - np.mean(x)) * (y - np.mean(y))) / (np.std(x) * np.std(y))


def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate ** 2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate ** 2)
    else:
        alpha = 0.
    return np.sum((correct - alpha * estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    B, M, N, C = correct.shape
    ssq = total = 0.
    for b in range(B):
        for c in range(C):
            for i in range(0, M - window_size + 1, window_shift):
                for j in range(0, N - window_size + 1, window_shift):
                    correct_curr = correct[b, i:i + window_size, j:j + window_size, c]
                    estimate_curr = estimate[b ,i:i + window_size, j:j + window_size, c]
                    ssq += ssq_error(correct_curr, estimate_curr)
                    total += np.sum(correct_curr ** 2)
    return ssq / total


def quality_assess(X, Y):
    
    Y = Y.permute(0, 2, 3, 1).cpu().numpy()  
    X = X.permute(0, 2, 3, 1).cpu().numpy()
    psnr = np.mean(cal_bwpsnr(Y, X)) 
    ssim = np.mean(cal_bwssim(Y, X))
    lmse = local_error(Y, X, 20, 10)
    ncc = compare_ncc(Y, X)
    return {'PSNR': psnr, 'SSIM': ssim, 'LMSE': lmse, 'NCC': ncc}
