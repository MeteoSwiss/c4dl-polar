import multiprocessing
import concurrent.futures

from numba import njit, prange
import numpy as np


def quick_cast(x, y):        
    num_threads = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        futures = {}
        limits = np.linspace(0, x.shape[0], num_threads+1).round().astype(int)
        def _cast(k0,k1):
            y[k0:k1,...] = x[k0:k1,...]        
        for k in range(len(limits)-1):
            args = (_cast, limits[k], limits[k+1])
            futures[executor.submit(*args)] = k
        concurrent.futures.wait(futures)


def cast(dtype=np.float16):
    xc = None
    def transform(raw):
        nonlocal xc
        if (xc is None) or (xc.shape != raw.shape):
            xc = np.empty_like(raw, dtype=dtype)
        quick_cast(raw, xc)
        return xc
    return transform


@njit(parallel=True)
def scale_array(in_arr, out_arr, scale):
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = scale[in_arr[i]]


def normalize(mean=0.0, std=1.0, dtype=np.float32):
    scaled = scaled_dt = None

    def transform(raw):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        normalize_array(raw, scaled, mean, std)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


def scale_log_norm(scale, threshold=None, missing_value=None,
    fill_value=0, mean=0.0, std=1.0, dtype=np.float32):

    log_scale = np.log10(scale).astype(np.float32)
    if threshold is not None:
        log_scale[log_scale < np.log10(threshold)] = np.log10(fill_value)
    if missing_value is not None:
        log_scale[missing_value] = np.log10(fill_value)
    log_scale[~np.isfinite(log_scale)] = np.log10(fill_value)
    log_scale -= mean
    log_scale /= std
    scaled = scaled_dt = None

    def transform(raw):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        scale_array(raw, scaled, log_scale)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


def scale_norm(scale, threshold=None, missing_value=None,
    fill_value=0, mean=0.0, std=1.0, dtype=np.float32):

    scale = scale.astype(np.float32).copy()
    scale[np.isnan(scale)] = fill_value
    if threshold is not None:
        scale[scale < threshold] = fill_value
    if missing_value is not None:
        missing_value = np.atleast_1d(missing_value)
        for m in missing_value:
            scale[m] = fill_value
    scale -= mean
    scale /= std
    scaled = scaled_dt = None    

    def transform(raw):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        scale_array(raw, scaled, scale)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


@njit(parallel=True)
def threshold_array(in_arr, out_arr, threshold):
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = np.float32(in_arr[i] >= threshold)


@njit(parallel=True)
def normalize_array(in_arr, out_arr, mean, std):
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = (in_arr[i]-mean)*inv_std


def R_threshold(scale, threshold):    
    thresholded = None
    scale_treshold = np.nanargmax(scale > threshold)

    def transform(rzc_raw):
        nonlocal thresholded
        if (thresholded is None) or (thresholded.shape != rzc_raw.shape):
            thresholded = np.empty_like(rzc_raw, dtype=np.float32)
        threshold_array(rzc_raw, thresholded, scale_treshold)

        return thresholded

    return transform

def transform_polar(mean=0,std=1,fill_value=0,lb=None,ub=None,dtype=np.float32):
    scaled = scaled_dt = None    
    def transform(raw_polar,raw_filter):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw_polar.shape):
            scaled = np.empty_like(raw_polar, dtype=np.float32)
            scaled_dt = np.empty_like(raw_polar, dtype=dtype)
        transform_polar_array(raw_polar,raw_filter,scaled,mean,std,lb,ub,fill_value)
        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt
    return transform

@njit(parallel=True)
def transform_polar_array(raw,filter,out_arr,mean,std,lb,ub,fill_value):
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    in_arr = replace(raw,filter,fill_value,lb,ub)
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = (in_arr[i]-mean)*inv_std

@njit(parallel=True)
def replace(raw,filter,fill_value,lb,ub):
    x = raw.ravel()
    filter = filter.ravel()
    x[np.isnan(x)] = fill_value
    x[(filter <= 2) | (filter >= 251)] = fill_value
    if lb is not None:
        x[x < lb] = fill_value
    if ub is not None:
        x[x > ub] = fill_value
    return x