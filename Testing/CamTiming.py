import numpy as np
import matplotlib.pyplot as plt

file = '.\\Testing\\CamTmSig.npz'

with np.load(file) as fh:
    data = fh['trig_sig']
    sampling_freq = fh['trig_fs']

# reshape and unpack the bits in the data
d = data.reshape(-1, 1)
d = np.unpackbits(d, 1)

# the 4th bit is trigger signal and the 3rd bit is exposure out signal
trig_sig = d[:, 3]
exp_out_sig = d[:, 2]
# 5-8 bits are 4 LED channels
led_sig = np.sum(d[:, 4:], 1)


# function to detect onset of trigger signal and exposure out signal
def findspikes(Vm0, thresh=-0.03, sampling_freq=20000):
    """
    find spike times in a whole cell recording
    :param vm: membrane potential traces
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
    for i in np.where(Vm0_th_idx_diff > 1)[0]:
        spk_region.append(tuple(Vm0_th_idx[idx_start:i + 1]))
        idx_start = i + 1
    # append last event
    spk_region.append(Vm0_th_idx[(i+1):])
    # spike times
    spt0 = [i[0] for i in spk_region]
    if sampling_freq is not None:
        spt0 = np.array(spt0) / sampling_freq
    return spt0


trig_on = findspikes(trig_sig, thresh=0.5, sampling_freq=None)
expout_on = findspikes(exp_out_sig, thresh=0.5, sampling_freq=None)
led_on = findspikes(led_sig, thresh=0.5, sampling_freq=None)
# trigger-exposure delay
size = min(trig_on.__len__(), expout_on.__len__())
TEDelay = np.array(expout_on[:size]) - np.array(trig_on[:size])
ELDelay = np.array(expout_on[:size]) - np.array(led_on[:size])
TLDelay = np.array(led_on[:size]) - np.array(trig_on[:size])
# epoching the exposure out signal based on trigger onset
period = trig_on[1] - trig_on[0]
offset = trig_on[0]
exp_out_epoch = np.zeros((period, trig_on.__len__()), dtype=exp_out_sig.dtype)
count = 0
for e in trig_on:
    try:
        exp_out_epoch[:, count] = exp_out_sig[e-offset:e+period-offset]
        count += 1
    except IndexError:
        pass
