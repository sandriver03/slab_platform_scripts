import Devices.TDTblackbox as tdt
import slab

import os
import numpy as np
import matplotlib.pyplot as plt


rcx_folder = "C:\\Users\\ch12gaxe\\Desktop\\RCX_testing"
rcx_name = 'RP2_FIR_coef.rcx'
rcx_file = os.path.join(rcx_folder, rcx_name)


rp2 = tdt.initialize_processor('RP2', 'USB', 1, rcx_file)

# generate a FIR filter coefficient using slab
filt = slab.Filter.band(samplerate=48828, length=16, frequency=3000, kind='lp')
# load filter coefficient into TDT
coefs = np.zeros(16*4, dtype=np.float32)
for i in range(4):
    coefs[i*16:(i + 1)*16] = filt.data.flatten() * (i + 1)
tdt.set_variable('fir_coef', coefs, rp2, 0)
# run
rp2.SoftTrg(1)
# read data
audio_ori = tdt.read_buffer(rp2, 'aud_ori', 0, 1000, float)
audio_filt = tdt.read_buffer(rp2, 'aud_filt', 0, 1000, float)
