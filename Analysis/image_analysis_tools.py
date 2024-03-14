from Analysis.image_polyROI import RoiPoly
from Analysis.image_phase_corr import phase_cross_correlation

import numpy as np
import tables as tl
from collections.abc import Sized
import scipy.ndimage as s_image
import logging
log = logging.getLogger()


def split_data_on_chs(img_data, tm_info, total_chs, channel_labels, remove_erray=False):
    """
    split multi-channel image series into individual channels
    Args:
        img_data: 3-D array of images, or 3-D tl.EArray, time in 1st dimension
        tm_info: 2-D array, (frame_Nr, frame_time). output from PrimeCam
        total_chs: int, number of total channels
        channel_labels: list or tuple or np.array, channel names; only used to create tl.earray
        remove_erray: Bool, if remove the original data if it is an earray

    Returns:
        img_indchs: list of 3-D array or tl.EArray, images from individual channel
        tm_indchs: list of 2-D array, (frame_Nr, frame_time)
        bad_frames: list of indexes, frames that are labelled in wrong channels
    """
    use_clustering = False
    idx_indchs = []
    bad_frames = []
    for ch_cnt in range(total_chs):
        img_idx = np.where((tm_info[:, 0] - tm_info[0, 0]) % total_chs == ch_cnt)[0]
        # check if there is any error in image indexing, by looking at average intensities for each frame
        f_avg = np.mean(img_data[img_idx, :, :], (1, 2))
        # mean is not a good indicator, should use median and quantile instead
        # f_avg_mean = f_avg.mean()
        # if a lot of frames are miss labeled, need to use clustering on all frames
        if np.diff(np.quantile(f_avg, [0.1, 0.9])) > 0.25 * np.median(f_avg):
            use_clustering = True
            idx_indchs = []
            log.warning("frame index mismatch. use histogram to separate different channels instead")
            break
        # only a couple of frames are miss labeled
        f_avg_median = np.median(f_avg)
        pass_idx = np.where(np.abs(f_avg - f_avg_median) < 0.25*f_avg_median)[0]
        idx_indchs.append(img_idx[pass_idx])
        bad_frames.append(np.where(np.abs(f_avg - f_avg_median) >= 0.25*f_avg_median)[0])

    if use_clustering:
        # use np.histogram and np.digitize to separate different channels
        f_avg = np.mean(img_data, (1, 2))
        bin_edges = np.histogram(f_avg, total_chs)[1]
        bin_edges[-1] = bin_edges[-1] + 1
        res = np.digitize(f_avg, bin_edges)
        # expectation for intensity: channel 0 > channel 1 > channel 2
        ch_orders = channel_labels.copy()
        ch_orders.sort()
        for ch in channel_labels:
            val_in_res = total_chs - ch_orders.index(ch)
            idx_indchs.append(np.where(res == val_in_res)[0])

    img_indchs = []
    tm_indchs = []
    # actually separating the images
    for ch_cnt in range(total_chs):
        img_idx = idx_indchs[ch_cnt]
        if isinstance(img_data, np.ndarray):
            img_indchs.append(img_data[img_idx, :, :])
        elif isinstance(img_data, tl.earray.EArray):
            img_indchs.append(_save_indch_images(img_data, img_idx[:, 0], channel_labels[ch_cnt]))
        else:
            raise ValueError('image data type: {} cannot be processed'.format(img_data.__class__))
        tm_indchs.append(tm_info[img_idx, :])

    # at this point, if image_data is earray and can be removed, then we should remove it
    if isinstance(img_data, tl.earray.EArray) and remove_erray:
        img_data.remove()

    return img_indchs, tm_indchs, bad_frames


def _save_indch_images(image_earray, img_idx_indch, channel_label):
    """
    save a subset of images in image_earray based on img_ids_indch to a new earray ordered by channel_label
    Args:
        image_earray: tl.earray.EArray
        img_idx_indch: tuple, list or np.ndarray, image indexes to be save
        channel_label: str, int, channel label

    Returns:
        image_indch_earray: tl.earray.EArray
    """
    name = image_earray._v_name + '_channel_{}'.format(str(channel_label))
    parent_path = image_earray._v_parent._v_pathname
    # create a new earray
    image_indch_earray = image_earray._v_file.create_earray(parent_path, name,
                                                            atom=tl.Atom.from_dtype(image_earray.dtype),
                                                            shape=(0, ) + image_earray.shape[1:])
    # copy the images indexed by img_idx_indch into the new earray
    for idx in img_idx_indch:
        image_indch_earray.append(image_earray[idx])

    return image_indch_earray


def detrend(data, axis=0, model='polynomial', order=1, keep_mean=True,
            copy=False, size_lim=2048, ram_lim=6144):
    """
    detread time series data with polynomial regression
    Args:
        data: data to be detrended, n-D np array, 1st dimension is time axis
        axis: int, along which axis to detrend
        model: string, model used to detrend the data. currently only 'polynomial'
        order: int, order of polynomial model
        keep_mean: bool, if keep the same mean after detrending
        copy: bool, if return a copy of the data without modifying original
        size_lim: float, largest block of data to be processed in one time, MB
        ram_lim: float, RAM usage limit in MB; maximum size of data to be operated in RAM

    Returns:
        nD np array of detrended data

    #TODO: take h5D earray as input and work on blocks
    """
    # data of > 3 dimensions are not supported
    if len(data.shape) > 3:
        raise ValueError('data with dimensions > 3 is not supported')
    # check the size of the data, and the type of the data
    _is_ndarray = False
    try:
        _nMB = data.nbytes / 1024 / 1024
        _is_ndarray = True
    except AttributeError:    # data is a h5d array
        _nMB = np.prod(data.shape) * data.dtype.itemsize / 1024 / 1024
    if _nMB / data.shape[1] < size_lim:
        _mode = 'vector'    # operating on one column in each iteration
    else:
        _mode = 'point'     # operating on a pixel-by-pixel base

    if copy:
        if _is_ndarray:
            result = data.copy()
        else:
            result = data[:]

    # use np.polyfit to do the detrending


# motion correction: not sure if needed
def select_roi(data, manual=False, **kwargs):
    """
    select ROI to perform motion correction
    Args:
        data: 2D np array, image data
        manual: if use manual selection. if not, use center 50% image to calculate registration
        kwargs: currently 'cmap' and 'clim' are used to control image display

    Returns:
        2D bool np array, image mask
        tuple, boundary points in x, y
    """
    if manual:
        roi = RoiPoly(data=data, **kwargs)
        mask = roi.get_mask()
        points = roi.get_boundary()
    else:
        mask = np.zeros(data.shape, dtype=bool)
        # set central 50% to true
        mask[int(data.shape[0]/4):int(3*data.shape[0]/4),
             int(data.shape[1]/4):int(3*data.shape[1]/4)] = 1
        low_p = int(data.shape[0]/4)
        high_p = int(3*data.shape[0]/4)
        xs = [low_p, high_p, high_p, low_p, low_p]
        ys = [low_p, low_p, high_p, high_p, low_p]
        points = (xs, ys)
    return mask, points


def calculate_shift_rigid(target_images, template, idx=None, mask=None, upsample_factor=100, ):
    # TODO: add option to run in multiprocessing cluster
    """
    calculate shifts between target images and one template assuming rigid transformation
    Args:
        target_images: array of image(s), image(s) to be shifted; 1st dimension is image index
        template: 2D np array, template image to be matched against
        idx: indexes for images in target_images to run calculation. if None calculate all images
        mask: 2D bool np array, mask image for template
        upsample_factor: up-sampling factor to get sub-pixel registration

    Returns:
        tuple of shifts in (frame_Nr, x, y) coordinates
    """
    stats = []
    if idx is None:
        idx = range(len(target_images))
    shifts = np.zeros([len(idx), 3])

    count = 0
    for id in idx:
        res = phase_cross_correlation(template, target_images[id], upsample_factor=upsample_factor,
                                      reference_mask=mask)
        shifts[count] = (id, *res[0])
        stats.append((res[1], res[2], ))
        count += 1
    return shifts, stats


def rigid_img_transform(images, shifts, idx=None, copy=False):
    # TODO: add option to run in multiprocessing cluster
    """
    perform rigid image transform (with scipy.ndimage.shift) to a serials of images. it will modify the data
    in place, unless copy is set to True
    Args:
        images: list, tuple or 3D np.array, series of images to be transformed
        shifts: list, tuple or 3D np.array, shifts for each images
        idx: iterable, if not None then the indexes of the images to be shifted according to shifts
        copy: if return a new copy of the data without modifying original data

    Returns:
        transformed images; in 3D np.array
    """
    if copy:
        if isinstance(images, np.ndarray):
            res = images.copy()
        elif isinstance(images, tl.earray.EArray):
            res = images[:]
        else:
            raise ValueError('data type: {} cannot be processed'.format(images.__class__))
    else:
        res = images
    if idx is None:
        idx = range(len(images))
    # check the format of shifts
    if len(shifts) < len(idx):
        if len(shifts) != 1:
            raise ValueError("number of shifts: {} does not match that of the images: {}".
                             format(len(shifts), len(res)))
        else:
            shifts = shifts * len(idx)

    # assuming always working along the 1st dimension
    for idx, shift in zip(idx, shifts):
        shifted = s_image.shift(res[idx], shift)
        res[idx] = shifted

    return res


def motion_correction(img_node, tm_data, reference_img, all_chs, ref_ch,
                      mask=None, upsample_factor=100):
    """
    perform motion correction on imaging data saved in .h5 file
    Args:
        img_node: tables.Earray, series of images to be transformed
        tm_data: np array of (2, -1), frame index and timing for each frame in img_node
        reference_img: np.array, frame to be used as reference
        all_chs: list or tuple, LED channels in the images
        ref_ch: int, channel to be used as reference
        mask: 'auto', 'manual' or np.array with same size as in img_node, mask used in masked correction
        upsample_factor: int, see calculate_shift_rigid
    Returns:

    """
    if isinstance(mask, np.ndarray) and mask.shape != img_node.shape[1:]:
        raise ValueError('mask size mismatch with those of images: {} vs {}'.
                         format(mask.shape, img_node.shape[1:]))
    if mask in ('auto', 'manual'):
        if mask == 'manual':
            mask = select_roi(reference_img, manual=True)
        else:
            mask = select_roi(reference_img, manual=False)

    # use reference channel to calculate shifts
    refch_idx = all_chs.index(ref_ch)
    refimg_idx = np.where((tm_data[:, 0] - tm_data[0, 0]) % len(all_chs) == refch_idx)[0]
    shifts = calculate_shift_rigid(img_node, reference_img, idx=refimg_idx, mask=mask,
                                   upsample_factor=upsample_factor)[0]
    # perform rigid transform on reference channels images
    img_node = rigid_img_transform(img_node, shifts, idx=refimg_idx)
    # for other channels, interpolate from 2 adjacent points in the reference channel
    for ch in all_chs:
        if ch != ref_ch:
            ch_idx = all_chs.index(ch)
            img_idx = np.where(tm_data[:, 0] % len(all_chs) == ch_idx)[0]
            # TODO: interpolate shifts from calculated shifts on the reference channel


def motion_correction_opticalflow(img_data, tm_data, reference_img, all_chs, ref_ch):
    """
    perform motion correction using optical flow method implemented by scikit-image
    Args:
        img_data: tables.Earray, np.ndarray or a list of those, series of images to be transformed
        tm_data: np array of (2, -1), or a list of this, frame index and timing for each frame in img_node
        reference_img: np.array, frame to be used as reference
        all_chs: list or tuple, LED channels in the images
        ref_ch: int, channel to be used as reference
    Returns:
        None, modifies inputs
    """
    # TODO


def _interpolate_shifts(shifts, tm_data, tot_chs, idx_cch):
    """
    interpolate shifts of current channel from calculated shifts on the reference channel
    Args:
        shifts: shifts calcualted from the reference channel
        tm_data: camera frame timing information for all channels
        tot_chs: int, number of channels in the data
        idx_cch: index of the frames for the current to be calculated channel

    Returns:
        shifts for current channel
    """
    # result in form of (idx, x, y)
    res = np.zeros([len(idx_cch), 3])
    for idx in idx_cch:
        # find the 2 frames immediately before and after current frame
        adj_frs = np.where(abs(idx - shifts[:, 0]) < tot_chs)
        # TODO


def image_spatial_filter(image_data, **filter_params):
    """
    spatial 2D filtering on image serials, either in .h5 file or in np.ndarray. assuming time is in
    the first dimension
    Args:
        image_data: tables.Earray or np.ndarray, series of images to be filtered
        filter_params:
            copy: bool, if create a separate copy for the result
            kernel: str, which filter kernel to use; other parameters depends on the kernel. currently
                    'gaussian', 'median' are implemented
            sigma: float, parameter for gaussian kernel, sigma of the 2-D gaussian
            size: int, parameter for median kernel, size of the filter window
    Returns:
        same dimension as original
    """
    if filter_params['kernel'].lower() == 'gaussian':
        param_keys = ('sigma', 'order', 'mode', 'cval', 'truncate')
        meth = s_image.gaussian_filter
    elif filter_params['kernel'].lower() == 'gaussian':
        param_keys = ('size', 'footprint', 'mode', 'cval', 'origin')
        meth = s_image.median_filter
    else:
        raise NotImplementedError('filtering with kernel: {} is not implemented'.
                                  format(filter_params['kernel']))
    meth_kwargs = {}
    for k in param_keys:
        if k in filter_params.keys():
            meth_kwargs[k] = filter_params[k]

    if 'copy' in filter_params.keys() and filter_params['copy']:
        if isinstance(image_data, np.ndarray):
            result = image_data.copy()
        elif isinstance(image_data, tl.earray.EArray):
            result = image_data[:]
        else:
            raise ValueError('image with class: {} cannot be processed'.
                             format(image_data.__class__))
    else:
        result = image_data

    for im_idx in range(result.shape[0]):
        result[im_idx] = meth(result[im_idx], **meth_kwargs)
    return result


# dF/F calculation
def deltaF_over_F(images, img_tms, LED_chs, channels_to_cal=None,
                  tm_wd=0., baseline_wd=3., dt=0.5):
    """
    calculate dF/F for given image series, with timing for each frame in img_tms

    Args:
        images: 3D np.array, image series (from individual channel)
        img_tms: 2D np.array, timing for each frame in the image series, in form of (frame_Nr, time), with unit second
        LED_chs: tuple or list or np.ndarray, LED channels in the image data
        channels_to_cal: list/tuple/array, which channels to calculate. channel idx starts from 0
        tm_wd: 2-element list/tuple/array, or float, timing window to be calculated, second
        baseline_wd: 2-element list/tuple/array, or float, images used to calculate base line (F), second
        dt: float, temporal bin size, second

    Returns:
        list of 3D np.array, dF/F values as function of time in individual channels
        list of 2D np.array, average intensity of each frame in individual channels, (frame_Nr, ch, avg)
    """
    if not isinstance(LED_chs, Sized):
        LED_chs = np.array([LED_chs])
    if not isinstance(LED_chs, np.ndarray):
        LED_chs = np.array(LED_chs)
    if channels_to_cal is None:
        channels_to_cal = LED_chs
    # check if channels_to_cal make sense
    if not isinstance(channels_to_cal, Sized):
        channels_to_cal = [channels_to_cal]
    if not isinstance(channels_to_cal, np.ndarray):
        channels_to_cal = np.array(channels_to_cal)
    if np.any(channels_to_cal > LED_chs):
        raise ValueError("parameter channenls_to_cal {} contains channel(s) that is unrealistic".
                         format(channels_to_cal))
    N_ledchs = len(LED_chs)

    if not isinstance(img_tms, np.ndarray):
        img_tms = np.array(img_tms)

    if not tm_wd:
        tm_wd = np.array([min(img_tms[:, 1]), max(img_tms[:, 1])])
    else:
        if not isinstance(tm_wd, Sized):
            tm_wd = [tm_wd]
        tm_wd = np.array(tm_wd)
        if len(tm_wd) == 1:
            tm_wd = (min(img_tms[:, 1]), min(img_tms[:, 1]) + tm_wd[0])
        elif tm_wd[1] < img_tms[0, 1]:
            # trial timing does not start from 0
            tm_wd = tm_wd + img_tms[0, 1]

    if not isinstance(baseline_wd, Sized):
        baseline_wd = [baseline_wd]
    baseline_wd = np.array(baseline_wd)
    if len(baseline_wd) == 1:
        baseline_wd = (min(img_tms[:, 1]), min(img_tms[:, 1]) + baseline_wd[0])
    elif baseline_wd[1] < img_tms[0, 1]:
        baseline_wd = baseline_wd + img_tms[0, 1]

    # generate time bins
    tm_bins = np.zeros(int(np.ceil((tm_wd[1] - tm_wd[0])/dt) + 1), dtype=float)
    temp = np.arange(tm_wd[0], tm_wd[1], dt, dtype=float)
    tm_bins[:len(temp)] = temp
    if len(temp) < len(tm_bins):
        tm_bins[-1] = temp[-1] + dt

    # take averages of images belong to each time bin, channel by channel
    # pre-allocating array
    dF_F = []
    F_avg = []  # keep track of average image intensity for each frame
    for ch in channels_to_cal:
        ch_idx = np.where(LED_chs == ch)[0]
        res = np.zeros((len(tm_bins)-1, ) + images.shape[1:])
        # calculate baseline
        bs_idx = np.where(np.logical_and((img_tms[:, 1] >= baseline_wd[0]) * (img_tms[:, 1] < baseline_wd[1]),
                                         (img_tms[:, 0] - img_tms[0, 0]) % N_ledchs == ch_idx))[0]
        bs = images[bs_idx, :, :].astype(float).mean(0)
        f_avg = []

        for bin_idx in range(len(tm_bins) - 1):
            img_idx = np.where(np.logical_and((img_tms[:, 1] >= tm_bins[bin_idx]) *
                                              (img_tms[:, 1] <  tm_bins[bin_idx + 1]),
                                              (img_tms[:, 0] - img_tms[0, 0]) % N_ledchs == ch_idx))[0]
            # take average of the images
            # assume 1st dimension is the time axis
            # numpy mean method use floats as intermediates
            # should only read the images once
            img_inbin = images[img_idx, :, :]
            res[bin_idx] = (img_inbin.mean(0) - bs) / bs
            # average intensity of each frame
            frame_info = np.zeros((len(img_idx), 3))
            frame_info[:, :2] = img_tms[img_idx, :]
            frame_info[:, 2] = img_inbin.mean((1, 2))
            f_avg.extend(frame_info)

        dF_F.append(res)
        F_avg.append(np.array(f_avg))

    return dF_F, F_avg


def downscale_image(img, downscale_factor=2, method=np.mean):
    """

    Args:
        img: 2D or 3D np array; if 3D 1st dimension is time (not a dimension of image)
        downscale_factor: int,
        method: np functions to be called to do the rescaling, e.g. np.mean, np.max, np.median

    Returns:
        downscaled image
    """
    # check image dimension
    pre_size = ()
    if len(img.shape) == 3:
        img_shape = img.shape[1:]
        pre_size = img.shape[:1]
    elif len(img.shape) == 2:
        img_shape = img.shape
    else:
        raise ValueError('image with shape: {} cannot be processed'.format(img.shape))

    # must be a full size rescaling
    if img_shape[0] % downscale_factor or img_shape[1] % downscale_factor:
        raise ValueError('image shape: {} cannot be fully divided by downscale factor: {}'
                         .format(img_shape, downscale_factor))

    # downscaling
    new_shape = (img_shape[0] // downscale_factor, downscale_factor,
                 img_shape[1] // downscale_factor, downscale_factor)
    temp_shape = pre_size + new_shape
    return method(img.reshape(temp_shape), (-1, -3)).astype(img.dtype)
