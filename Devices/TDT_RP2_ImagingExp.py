# light controller with TDT RP2 device
from Core.Device import Device
from Core.Setting import DeviceSetting
from Config import get_config

from traits.api import CFloat, Str, CInt, CArray, Any, Property, List, \
    TraitHandler, Trait, Dict, Instance, cached_property
from Devices import TDTblackbox as tdt
import numpy as np
import os
import logging
log = logging.getLogger()


# TODO: LED delay can be reduced by 6 ticks (when running in 50 kHz)
class RP2ImagingControllerSetting(DeviceSetting):
    """
    channel mapping
        uses digital output channels 0-3; channel 0 is used for camera trigger, and channels 1-3 are used to
        control up to 3 light sources. channel 0 will always output all the pulses, and channel 1-3 will be
        alternating when enabled
    pulse_width
        100 is full width, and 0 is none. pulse width does not affect channel 0
    """

    pulse_freq = CFloat(30, group='primary', dsec='frequency of the pulse, Hz', reinit=False)
    pl_ch0 = CFloat(1, group='primary',
                   dsec='length of the pulse on channel 0; ms', reinit=False)
    pl_ch1 = CFloat(1, group='primary',
                   dsec='length of the pulse on channel 1; ms', reinit=False)
    pl_ch2 = CFloat(1, group='primary',
                   dsec='length of the pulse on channel 2; ms', reinit=False)
    pl_ch3 = CFloat(1, group='primary',
                   dsec='length of the pulse on channel 3; ms', reinit=False)
    channels_touse = List([0, 1, 2, 3], group='primary',
                            dsec='which output channels to use, should be in 0-3', reinit=False)
    n_channels = CInt(4, group='status', dsec='total number of output channels, n')
    device_freq = CFloat(48828.1250, group='status', dsec='sampling frequency of the device (Hz)')
    rcx_file = Str('RCX\\RP2_Tonotopy_1.rcx', group='status', dsec='name of the rcx file to load')
    processor = Str('RP2', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    light_ch_delays = CFloat(9.95, group='status', dsec='delay between onset of frame trigger to all rows in exposure'
                                                        ' for the Prime camera, ms')
    device_type = 'ImagingExp_RP2'

    # intermediate variables
    pulse_period_n = Property(CInt, depends_on=['pulse_freq', 'device_freq'],
                              group='derived', dsec='period of the pulse, n samples')
    chs_delay_n = Property(CArray, depends_on=['pulse_freq', 'device_freq', 'light_ch_delays'],
                         group='derived', dsec='delays on channel 0-3, n samples')
    chs_ld_n = Property(CArray, depends_on=['pulse_freq', 'pld_all', 'device_freq', 'light_ch_delays'],
                      group='derived', dsec='length of each pulse on channels in use, n samples')
    pld_all = Property(CArray, depends_on=['pl_ch0', 'pl_ch1', 'pl_ch2', 'pl_ch3'],
                       group='status', dsec='length of each pulse on channel 0-3, ms')
    n_chs_inuse = Property(CInt, depends_on=['channels_touse'], group='derived',
                           dsec='number of channels in use')

    @cached_property
    def _get_pulse_period_n(self):
        return int(1 / self.pulse_freq * self.device_freq)

    @cached_property
    def _get_pld_all(self):
        return np.array([self.pl_ch0, self.pl_ch1, self.pl_ch2, self.pl_ch3]).ravel()

    @cached_property
    def _get_chs_delay_n(self):
        delays_n = int(self.light_ch_delays / 1000 * self.device_freq)
        delays_all = np.ones(self.n_channels, dtype=int) * delays_n
        delays_all[delays_all == 0] = 1
        return delays_all

    @cached_property
    def _get_chs_ld_n(self):
        # maximum pulse length in ms
        max_pl = 1 / self.pulse_freq * 1000
        self.pld_all[self.pld_all > max_pl] = max_pl
        wd = np.int32(self.pld_all * self.device_freq / 1000)
        # minimum pulse length is at least 1 point
        wd[wd == 0] = 1
        return wd

    @cached_property
    def _get_n_chs_inuse(self):
        return len(self.channels_touse)


class RP2ImagingController(Device):
    """
    the buffer 'PulseTTL' will not reset when calling pause/start. to reset the buffer, need to
    send software trigger 2 to the circuit, or use method reset_buffer
    """
    setting = RP2ImagingControllerSetting()
    buffer = Any
    handle = Any
    _use_default_thread = False

    ParaTag_mapping = Dict({'pulse_period': 'pulseT_n',
                            'chs_delay': ['Ch0del', 'Ch1del', 'Ch2del', 'Ch3del'],
                            'chs_ld': ['Ch0wd', 'Ch1wd', 'Ch2wd', 'Ch3wd'],
                            'chs_shift': ['Ch0Shift', 'Ch1Shift', 'Ch2Shift', 'Ch3Shift'],
                            },)
    _DOut_ch_shift = List([1, 2, 4, 8])

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor,
                                               connection=self.setting.connection,
                                               index=1,
                                               path=os.path.join(expdir, self.setting.rcx_file))
        self.handle.Run()
        TDT_freq = self.handle.GetSFreq()
        if abs(TDT_freq - self.setting.device_freq) > 1:
            log.warning('TDT sampling frequency is different from that specified in software: {} vs. {}'.
                        format(TDT_freq, self.setting.device_freq))
            # self.setting.device_freq = self.handle.GetSFreq()
        self._output_specs = {'type': 'digital_signal',
                              'shape': (-1, 1),
                              'sampling_freq': self.setting.device_freq,
                              'dtype': np.uint8,
                              }
        # ch delays should only need to be set once
        for idx in range(self.setting.n_channels):
            self.handle.SetTagVal(self.ParaTag_mapping['chs_delay'][idx], self.setting.chs_delay_n[idx])

    def _configure(self, **kwargs):
        # TODO: this is a slow implementation; should only set the params that are changed
        # configure pulse period
        if 'pulse_freq' in self._changed_params:
            self.handle.SetTagVal('pulseT_n', self.setting.pulse_period_n)
        # configure number of channels in use
        if 'channels_touse' in self._changed_params:
            self.handle.SetTagVal('N_ActCh', self.setting.n_chs_inuse)
        # configure pulse width and channel bit shift (used to target different channel)
        for idx in range(self.setting.n_chs_inuse):
            ch = self.setting.channels_touse[idx]
            self.handle.SetTagVal(self.ParaTag_mapping['chs_ld'][idx], self.setting.chs_ld_n[ch])
            self.handle.SetTagVal(self.ParaTag_mapping['chs_shift'][idx], self._DOut_ch_shift[ch])

    def _pause(self):
        # no way to pause the device until it finishes
        self.handle.SoftTrg(1)

    def _start(self):
        self.handle.SoftTrg(1)

    def read(self, start_idx=0, nsamples=None, dtype=np.uint8):
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
            nsamples = self.handle.GetTagVal('PulseTTL_i')
        # the ReadTagV treats each 32-bit buffer as floats
        data = np.array(self.handle.ReadTagV('PulseTTL', start_idx, nsamples), dtype=np.float32)
        # use np.ndarray.view to change the representation of dtype
        if np.issubdtype(dtype, np.integer):
            data = data.view(np.int32).astype(dtype)
        # data is 1 by n, should be uint8
        return data.reshape(-1, 1)

    def _stop(self):
        self.handle.Halt()


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

    ld = RP2ImagingController()
    # fd.setting.configure_traits()
