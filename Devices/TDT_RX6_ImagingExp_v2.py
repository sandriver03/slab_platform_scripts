# light controller with TDT RX6 device
# use a different .rcx file that allows continuous stimulation in one trial
from Core.Device import Device
from Core.Setting import DeviceSetting
from Config import get_config

from traits.api import CFloat, Str, CInt, CArray, Any, Property, List, \
    TraitHandler, Trait, Dict, Instance, cached_property, Enum
from Devices import TDTblackbox as tdt
import numpy as np
import os
import logging
log = logging.getLogger()

PARAMS_VALIDATE = ('burst_len', 'ISI_burst', 'n_burst', 'ISI')


class RX6ImagingControllerSetting(DeviceSetting):
    """
    channel mapping
        uses digital output channels 0-3; channel 0 is used for camera trigger, and channels 1-3 are used to
        control up to 3 light sources. channel 0 will always output all the pulses, and channel 1-3 will be
        alternating when enabled
    pulse_width
        100 is full width, and 0 is none. pulse width does not affect channel 0
    """
    # TODO: since there is always a delay between light on and camera all rows exposure, there will be a upper limit for
    #  light pulse duration, depending on the pulse_freq and camera exposure time
    # TODO: put camera exposure here as a state variable

    # parameters related to light/camera control
    # parameter tag specifications:
    #   tdt_param: actual values need to be set in the TDT circuit
    #   tag_name: name of the TDT param tag need to be set
    #   special: this parameter is enclosed in a list
    pulse_freq = CFloat(30, group='primary', dsec='frequency of the pulse, Hz', reinit=False,
                        tdt_param='pulse_period_n')
    pl_ch0 = CFloat(1, group='primary',
                    dsec='length of the pulse on channel 0; ms', reinit=False, special=True)
    pl_ch1 = CFloat(1, group='primary',
                    dsec='length of the pulse on channel 1; ms', reinit=False, special=True)
    pl_ch2 = CFloat(1, group='primary',
                    dsec='length of the pulse on channel 2; ms', reinit=False, special=True)
    pl_ch3 = CFloat(1, group='primary',
                    dsec='length of the pulse on channel 3; ms', reinit=False, special=True)
    channels_touse = List([0, 1, 2, 3], group='primary',
                          dsec='which output channels to use, should be in 0-3', reinit=False,
                          special=True)
    n_channels = CInt(4, group='status', dsec='total number of output channels, n')
    device_freq = CFloat(97656.2500, group='status', dsec='sampling frequency of the device (Hz)')
    rcx_file = Str('RCX\\RX6_Tonotopy_Dred_Full.rcx', group='status', dsec='name of the rcx file to load')
    processor = Str('RX6', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    light_ch_delays = CFloat(9.95, group='status', dsec='delay between onset of frame trigger to all rows in exposure'
                                                        ' for the Prime camera, ms')
    device_type = 'ImagingExp_RX6'

    # intermediate variables
    pulse_period_n = Property(CInt, depends_on=['pulse_freq', 'device_freq'],
                              group='derived', dsec='period of the pulse, n samples',
                              tag_name='pulseT_n')
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

    # parameters related to sound generation
    # IMPORTANT: the first value in the WN_amps, tone_amps and tone_freqs is not used. Handling of this issue is
    # expected to be carried out in the Experiment classes using this device
    WN_amps = CArray(dtype=np.float32, group='primary', dsec='array of white noise amplitudes in one trial',
                     reinit=False, tag_name='WNamp_seq')
    tone_amps = CArray(dtype=np.float32, group='primary', dsec='array of tone amplitudes in one trial',
                       reinit=False, tag_name='toneamp_seq')
    tone_freqs = CArray(dtype=np.float32, group='primary', dsec='array of tone frequencies in one trial',
                        reinit=False, tag_name='tonefreq_seq')
    N_stims = CInt(10, group='primary', dsec='number of stimuli', reinit=False, tag_name='N_stim')
    stimOn_delay = CFloat(3., group='primary', dsec='delay of the first stimulus', reinit=False,
                          tdt_param='stim_delay_n')
    ISI = CFloat(3., group='primary', dsec='inter stimulus interval, s', reinit=False, tdt_param='ISI_n')
    n_burst = CInt(5, group='primary', dsec='number of bursts in each stimulus', reinit=False, tag_name='N_bst')
    ISI_burst = CFloat(0.1, group='primary', dsec='inter stimulus interval for each sound in the burst, s',
                       reinit=False, tdt_param='ISI_bst_n')
    burst_len = CFloat(0.05, group='primary', dsec='length of each sound in the burst, s',
                       reinit=False, tdt_param='burst_len_n')
    use_noise = Enum(0, 1, group='primary', dsec='if use the white noise stimulation',
                     reinit=False, tag_name='use_noise')
    use_tone = Enum(0, 1, group='primary', dsec='if use the pure tone stimulation',
                    reinit=False, tag_name='use_tone')
    # intermediate variables; mostly time -> N datapoint, to be used in TDT
    # all of them depend on device frequency. however the device frequency should not be changed
    stim_delay_n = Property(CInt, depends_on=['stim_delay_n'], group='status', dsec='number of data points',
                            tag_name='stim_delay_n')
    ISI_n = Property(CInt, depends_on=['ISI'], group='status', dsec='number of data points',
                     tag_name='ISI_n')
    ISI_burst_n = Property(CInt, depends_on=['ISI_burst'], dsec='number of data points',
                           tag_name='ISI_bst_n')
    burst_len_n = Property(CInt, depends_on=['burst_len'], dsec='number of data points',
                           tag_name='burst_len_n')

    @cached_property
    def _get_stim_delay_n(self):
        return int(self.stim_delay * self.device_freq)

    @cached_property
    def _get_ISI_n(self):
        return int(self.ISI * self.device_freq)

    @cached_property
    def _get_ISI_burst_n(self):
        return int(self.ISI_burst * self.device_freq)

    @cached_property
    def _get_burst_len_n(self):
        return int(self.burst_len * self.device_freq)

    # should use _configure_validation in the Device class to validate timing parameters
    # the total burst length (n_burst * ISI_burst) must be shorter than ISI
    # individual burst length (burst_len) must be shorter than burst ISI (ISI_burst)


class RX6ImagingController(Device):
    """
    the buffer 'PulseTTL' will not reset when calling pause/start. to reset the buffer, need to
    send software trigger 2 to the circuit, or use method reset_buffer
    """
    setting = RX6ImagingControllerSetting()
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
        # use ZBus in this case
        self.ZBus = tdt.initialize_zbus(connection=self.setting.connection)
        self.handle.Run()
        TDT_freq = self.handle.GetSFreq()  # not necessarily returns correct value
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

    def _configure_validation(self, **kwargs):
        if set(kwargs.keys()).intersection(PARAMS_VALIDATE):
            params = self._get_validate_params(**kwargs)
            # individual burst length (burst_len) must be shorter than burst ISI (ISI_burst)
            if params['burst_len'] >= params['ISI_burst']:
                msg = "The burst length: {} is longer than the maximum allowed by burst ISI: {}".\
                    format(params['burst_len'], params['ISI_burst'])
                print(msg)
                log.warning(msg)
            # the total burst length (n_burst * ISI_burst) must be shorter than ISI
            if params['burst_len'] * params['ISI_burst'] >= params['ISI']:
                msg = "The length of each burst: {} is longer than the maximum allowed by ISI: {}". \
                    format(params['burst_len'] * params['ISI_burst'], params['ISI'])
                print(msg)
                log.warning(msg)

    def _get_validate_params(self,  **kwargs):
        param_dict = {}
        for p in PARAMS_VALIDATE:
            if p in kwargs:
                param_dict[p] = kwargs[p]
            else:
                param_dict[p] = getattr(self.setting, p)
        return param_dict

    def _configure(self, **kwargs):
        set_chs = False
        for param in self._changed_params:
            par_trait = self.setting.trait(param)
            if par_trait.tag_name:
                # set the value of current trait into tag_name on TDT
                self.handle.SetTagVal(par_trait.tag_name, getattr(self.setting, param))
            if par_trait.tdt_param:
                # TDT tag value is stored in par_trait.tdt_param
                tdt_param_name = par_trait.tdt_param
                tag_trait = self.setting.trait(tdt_param_name)
                self.handle.SetTagVal(tag_trait.tag_name, getattr(self.setting, tdt_param_name))
            if par_trait.special:
                set_chs = True

        if set_chs:
            # configure number of channels in use
            if 'channels_touse' in self._changed_params:
                self.handle.SetTagVal('N_ActCh', self.setting.n_chs_inuse)
            # configure pulse width and channel bit shift (used to target different channel)
            for idx in range(self.setting.n_chs_inuse):
                ch = self.setting.channels_touse[idx]
                self.handle.SetTagVal(self.ParaTag_mapping['chs_ld'][idx], self.setting.chs_ld_n[ch])
                self.handle.SetTagVal(self.ParaTag_mapping['chs_shift'][idx], self._DOut_ch_shift[ch])

    def _pause(self):
        # ZBus trigger: (racknum, trigtype, delay (ms)); 0 = all racks; 0 = pulse, 1 = high, 2 = low; minimum of 2
        self.ZBus.zBusTrigA(0, 0, 4)

    def _start(self):
        self.ZBus.zBusTrigA(0, 0, 4)

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
            data = data.view(dtype).astype(dtype)
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

    ld = RX6ImagingController()
    # fd.setting.configure_traits()
