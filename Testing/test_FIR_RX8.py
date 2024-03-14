import Devices.TDTblackbox as tdt
import slab

import os
import numpy as np
import matplotlib.pyplot as plt


def shuffle_filter_coefs(coefs_array, dtype=np.float32):
    """
    shuffle the filter coefficient array, so it can be used in the TDT FIR2 component
    see TDT RPvdsEx manual for the details
    :param coefs_array: 2d coefficient array, 2nd dimension is channel
    :param dtype: data type of returned array
    :return: shuffled and flattened coefficient array
    """
    if len(coefs_array.shape) < 2:
        coefs_array = coefs_array.reshape(-1, 1)
    # 1st dimension is the taps and 2nd the channels
    n_taps = coefs_array.shape[0]
    if n_taps % 2 != 0:
        raise ValueError("the nTaps of the filters must be even")
    # set data type to float32, which is used in TDT
    coefs_shuffled = np.zeros(coefs_array.shape, dtype=dtype)
    coefs_shuffled[::2, :] = coefs_array[:int(n_taps/2), :]
    coefs_shuffled[1::2, :] = coefs_array[int(n_taps/2):, :]
    return coefs_shuffled.ravel(order='F')


rcx_folder = "C:\\Users\\PC\\Desktop"
rcx_name = 'RX8_FIR_test.rcx'
rcx_file = os.path.join(rcx_folder, rcx_name)


rx8 = tdt.initialize_processor('RX8', 'GB', 1, rcx_file)
zbus = tdt.initialize_zbus(connection='GB')

# generate a FIR filter coefficient using slab
filt1 = slab.Filter.band(samplerate=97656, length=128, frequency=200, kind='lp')
filt2 = slab.Filter.band(samplerate=97656, length=128, frequency=(400, 800), kind='bp')
filt3 = slab.Filter.band(samplerate=97656, length=128, frequency=(1000, 1800), kind='bp')
filt4 = slab.Filter.band(samplerate=97656, length=128, frequency=3000, kind='hp')
# load filter coefficient into TDT
coefs = np.zeros((128, 4), dtype=np.float32)
coefs[:, 0] = filt1.data.flatten()
coefs[:, 1] = filt2.data.flatten()
coefs[:, 2] = filt3.data.flatten()
coefs[:, 3] = filt4.data.flatten()
coefs = shuffle_filter_coefs(coefs)
tdt.set_variable('filter_coefs', coefs, rx8, 0)


# run
zbus.zBusTrigA(0, 1, 3)

# pause
zbus.zBusTrigA(0, 2, 3)

# use soft1 to trigger 1s recordings
rx8.SoftTrg(1)

# read data, after recording
audio_ori = slab.Sound(tdt.read_buffer(rx8, 'aud', 0, 97656, float), samplerate=97656)
audio_filt = slab.Sound(tdt.read_buffer(rx8, 'aud_filt', 0, 97656, float), samplerate=97656)
