# light controller with TDT RP2 device
from Core.Device import Device
from Core.Setting import DeviceSetting
from Config import get_config
# import slab   # use slab to generate stimulus

from traits.api import CFloat, Str, CInt, CArray, Any, Property, List, Float, \
    TraitHandler, Trait, Dict, Instance, cached_property
from Devices import TDTblackbox as tdt
import numpy as np
import os

import logging
log = logging.getLogger()


class RX6SpeakerCalSetting(DeviceSetting):
    """
    channel mapping
        uses digital output channels 0-3; channel 0 is used for camera trigger, and channels 1-3 are used to
        control up to 3 light sources. channel 0 will always output all the pulses, and channel 1-3 will be
        alternating when enabled
    pulse_width
        100 is full width, and 0 is none. pulse width does not affect channel 0
    """
    stim_length = Float(1, group='primary', dsec='length of stimulus, second', reinit=False)

    device_freq = CFloat(97656.2500, group='status', dsec='sampling frequency of the device (Hz)')
    rcx_file = Str('RCX\\RX6_speaker_calibration.rcx', group='status', dsec='name of the rcx file to load')
    processor = Str('RX6', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    max_stim_length_n = CInt(1000000, group='status', dsec='maximum length for stimulus in number of data points')

    # derived paramters
    stim_length_n = Property(CInt, depends_on=['stim_length', 'device_freq'])

    device_type = 'SpeakerCal_RX6'

    @cached_property
    def _get_stim_length_n(self):
        stim_n = int(self.stim_length * self.device_freq)
        if stim_n > self.max_stim_length_n:
            log.warning("stimlulus length: {} is longer than maximum length allowed: {}. "
                        "Stimulus is capped".format(stim_n, self.max_stim_length_n))
            stim_n = self.max_stim_length_n
        return stim_n


class RX6SpeakerCal(Device):
    """
    the buffer 'PulseTTL' will not reset when calling pause/start. to reset the buffer, need to
    send software trigger 2 to the circuit, or use method reset_buffer
    """
    setting = RX6SpeakerCalSetting()
    buffer = Any
    handle = Any
    _use_default_thread = False

    stimulus = Any

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor,
                                               connection=self.setting.connection,
                                               index=1,
                                               path=os.path.join(expdir, self.setting.rcx_file))
        self.handle.Run()
        TDT_freq = self.handle.GetSFreq()  # not necessarily returns correct value
        if abs(TDT_freq - self.setting.device_freq) > 1:
            log.warning('TDT sampling frequency is different from that specified in software: {} vs. {}'.
                        format(TDT_freq, self.setting.device_freq))
            # self.setting.device_freq = self.handle.GetSFreq()
        self._output_specs = {'type': 'analog_signal',
                              'shape': (-1, 1),
                              'sampling_freq': self.setting.device_freq,
                              'dtype': np.float32,
                              }

    def _configure(self, **kwargs):
        # configure stimulus length
        if 'stim_length' in self._changed_params:
            # set stimulus buffer length
            self.handle.SetTagVal('stimBufLen_n', self.setting.stim_length_n)
            # set recording buffer length
            self.handle.SetTagVal('recBufLen_n', self.setting.stim_length_n + 200)

    def _pause(self):
        # no way to pause the device until it finishes
        pass

    def _start(self):
        self.handle.SoftTrg(1)

    def read(self, start_idx=0, nsamples=None, dtype=np.float32):
        """
        Read TTL signal from the PulseTTL buffer
        the TTL signal is read as 8 bit unsigned integers. unpack it into 8 bit to get 8 channels
        Only 6 channels are used currently.
        Args:
            start_idx: int, starting index of the read
            nsamples: number of samples to read. if None, read all available samples
            dtype: data type of the result, currently only works with numpy types
        Returns:
            array of uint8, unpack it to get 8 channel TTLs
            significant digits seem to be flipped between python and TDT
            ***8-by-N array, TTL from 8 channels
        """
        if not nsamples:
            nsamples = self.handle.GetTagVal('recBuf_idx')
        # the ReadTagV treats each 32-bit buffer as floats
        data = np.array(self.handle.ReadTagV('audio_data', start_idx, nsamples), dtype=np.float32)
        # use np.ndarray.view to change the representation of dtype
        if np.issubdtype(dtype, np.integer):
            data = data.view(np.int32).astype(dtype)
        # data is 1 by n, should be uint8
        return data.reshape(-1, 1)

    def _stop(self):
        self.handle.Halt()

    def load_stimulus(self):
        """
        load stimulus into tdt buffer. stimulus should be prepared with slab toolbox
        Returns:
            None
        """
        if not self.stimulus:
            log.error('stimulus is not prepared. generate it using Slab toolbox')
        if self.stimulus.samplerate != self.setting.device_freq:
            log.error("stimulus sampling freq: {} is not the same as device sampling freq: {}".
                      format(self.stimulus.samplerate, self.setting.device_freq))
        data_length = self.stimulus.nsamples
        if self.stimulus.nsamples > self.setting.max_stim_length_n:
            log.warning("stimulus length: {} is longer than maximum length allowed: {}. stimulus is chopped.".
                        format(self.stimulus.nsamples, self.setting.max_stim_length_n))
            data_length = self.setting.max_stim_length_n
        self.handle.WriteTag('stim_data',  0, self.stimulus.data[:data_length])


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    ld = RX6SpeakerCal()
    # fd.setting.configure_traits()
