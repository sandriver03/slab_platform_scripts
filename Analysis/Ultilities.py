from .image_analysis_tools import downscale_image

import numpy as np
from collections.abc import Sized

RAM_LIMIT = 6  # GB


def _prepare_data(data, copy=False, ram_lim=RAM_LIMIT, down_scale=1, temp_path='/'):
    """
    check if data is a file or a memory in form of numpy.ndarray, check the size of the data
    Args:
        data: data to be checked, np.ndarray or tables.EArray
        copy: bool, if to make a copy of the data if the data is np.ndarray
        ram_lim: float, maximum allowed size of data in ram, GB
        down_scale: int, if downscaling the image by an integer factor
        temp_path: str, where a temporal tables.earray should be created
    Returns:
        data: handle to the data, np.ndarray or tables.EArray,
        _is_ndarray: bool
    """
    # data of > 3 dimensions are not supported
    if len(data.shape) > 3:
        raise ValueError('data with dimensions > 3 is not supported')
    # check the size of the data, and the type of the data
    _is_ndarray = False
    try:
        _nMB = data.nbytes / 1024 ** 2
        _is_ndarray = True
    except AttributeError:    # data is a h5d array
        _nMB = np.prod(data.shape) * data.dtype.itemsize / 1024 ** 2

    # apply downscaling
    _nMB = _nMB / down_scale

    if copy and _is_ndarray and _nMB * 2 / 1024 < ram_lim:
        if down_scale > 1:
            data = downscale_image(data, down_scale)
        else:
            data = data.copy()
    if not _is_ndarray:
        if _nMB / 1024 < ram_lim:
            data = data[:]
            if down_scale > 1:
                data = downscale_image(data, down_scale)
            _is_ndarray = True
        else:
            # copy the data into a temporal earray in the temp group
            if down_scale == 1:
                data = data._v_file.copy_node(data._v_pathname, temp_path)
            else:
                # create a new earray
                new_shape = np.array(data.shape, dtype=np.int)
                new_shape[0] = -1
                new_shape[1:] = new_shape[1:] // down_scale
                new_data = data._v_file.create_earray(temp_path, data.name, atom=data.atom,
                                                       shape=tuple(new_shape))
                # read the image one by one, downscale, and write into new node
                for img in data:
                    new_data.append(downscale_image(img, down_scale))
                data = new_data
            data._v_file.flush()

    return data, _is_ndarray


def findspikes(Vm0, thresh=-0.03, sampling_freq=None):
    """
    find spike times in a whole cell recording; here is used to detect onset of trigger signal and exposure out signal
    :param Vm0: membrane potential traces
    :param thresh: threshold to detect spikes
    :param sampling_freq: sampling frequency of vm
    :return: numpy array of threshold rising edge crossing time, in second, or idx if sf is None
    """
    # find regions contain spike
    Vm0_th_idx = np.where(Vm0 > thresh)[0]  # indexes above threshold
    # get indexes of the regions
    spk_region = []
    Vm0_th_idx_diff = np.diff(Vm0_th_idx)
    idx_start = 0
    if np.any(Vm0_th_idx_diff):
        for i in np.where(Vm0_th_idx_diff > 1)[0]:
            spk_region.append(tuple(Vm0_th_idx[idx_start:i + 1]))
            idx_start = i + 1
        # append last event
        spk_region.append(Vm0_th_idx[(i+1):])
    # spike times
    spt0 = [i[0] for i in spk_region]
    # spike width
    sw = [i.__len__() for i in spk_region]
    if sampling_freq is not None:
        spt0 = np.array(spt0) / sampling_freq
    return spt0, sw


# convert the unit of timing from the camera to different units
def primecam_convert_timing_unit(timing_data, target_unit='s', copy=False):
    """
    convert the unit of timing from the camera to different units
    Args:
        timing_data: array, timing data from the camera
        target_unit: string, target timing unit to be converted to
        copy: boolean, if return a copy of the input data. otherwise original is modified

    Returns:
        converted array of timing data
    """
    if target_unit.lower() == 's':
        scale_factor = 1e-4
    elif target_unit.lower() == 'ms':
        scale_factor = 1e-1
    else:
        raise ValueError('convert to unit {} is not implemented'.format(target_unit))

    if not isinstance(timing_data, Sized):  # a single number
        timing_data = np.array([timing_data])
    if isinstance(timing_data, list):  # directly coming from camera is list
        timing_data = np.array(timing_data)
    if copy:
        timing_data = timing_data.copy()
    if len(timing_data.shape) == 2 and timing_data.shape[1] == 2:
        # data directly coming from camera
        timing_data[:, 1] = timing_data[:, 1] * scale_factor
    else:
        # otherwise, assume all data points are timing data
        timing_data = timing_data * scale_factor
    return timing_data


def get_frame_timing(di_array, sampling_freq=None):
    """
    get trigger and frame timing from recorded digital channels
    currently, the 8 digital channels are ordered as below:
        0-1: not used
        2: exposure out signal from the PRIME camera
        3: trigger signal used to trigger each frame
        4-7: control signal for 4 LEDs (currently 3 used), arranged backwards, i.e. 7 is the 0th channel
    :param di_array: should be (N, ) uint8 array
    :param sampling_freq: Hz, sampling frequency of the digital array
    :return: list of lists, index of frames for each LED channel
    """
    # unpack uint8 array to get the 8 different digital channels
    dc_sig = di_array.reshape(-1, 1)
    dc_sig = np.unpackbits(dc_sig, 1)
    # get timing on each meaningful channel
    trig_on = findspikes(dc_sig[:, 3], 0.5, sampling_freq)
    expo_out = findspikes(dc_sig[:, 2], 0.5, sampling_freq)
    frames_byLED = []
    for i in range(7, 3, -1):
        # LED_sig.append(findspikes(dc_sig[:, i], 0.5))
        # check the overlapping between exposure output from the camera and LED trigger
        LED_expo_overlap = findspikes(dc_sig[:, i] + dc_sig[:, 2], 1.5)
        # get index of frames when this LED was used
        start_frame_idx = 0
        frames_thisLED = []
        for start_idx, width in zip(LED_expo_overlap[0], LED_expo_overlap[1]):
            # only check when LED was actually on
            if width < 1:
                continue
            # frames are ordered, so just need to continue search from last found frame
            for frame_idx in range(start_frame_idx, len(expo_out[0])):
                if start_idx in range(expo_out[0][frame_idx] - 1, expo_out[0][frame_idx] + expo_out[1][frame_idx]):
                    frames_thisLED.append(frame_idx)
                    start_frame_idx = frame_idx
                    break

        frames_byLED.append(frames_thisLED)

    return frames_byLED
