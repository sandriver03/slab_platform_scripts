# using the labplatform environment
# testing the RX6RX8 for imaging setup
from Devices import TDTblackbox as tdt
from Devices.TDT_RX6RX8_Imaging import RX6RX8ImagingController
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import logging


log = logging.getLogger()
log.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)

ld = RX6RX8ImagingController()


# testing synchronization
# using white noise,
stim_seq = [0.1, 0.5, 1]*10
adap_seq = [1, 0.5, 0.1]*10
ld.configure(use_noise=1, use_adaptor=1, use_tone=0, WN_amps=[0, ] + stim_seq, adap_amps=[0, ] + adap_seq,
             n_burst=1, N_stims=len(stim_seq), ISI=3., burst_len=0.2, prob_delay=0.75, ISI_burst=1)

ld.configure(use_noise=1, use_adaptor=1, use_tone=0, WN_amps=[0, ] + stim_seq, n_burst=1,
             N_stims=len(stim_seq), ISI=20, burst_len=0.2, prob_delay=0.75, ISI_burst=1)

t0 = time.time()
ld.start()
run_time = 610   # running time
while time.time() - t0 < run_time:
    time.sleep(0.1)
ld.pause()

# read data from buffer
# first index seems to be 0
# nsamples = rp2.GetTagVal('tone_idx')
# tone_data = tdt.read_buffer(rp2, 'tone_data', 0, nsamples, np.float32)
# ttl_data = tdt.read_buffer(rp2, 'TTL_data', 0, nsamples, np.float32)

# plot figure
# plt.figure()
# plt.plot(tone_data)
