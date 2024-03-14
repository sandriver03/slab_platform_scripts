# light controller with TDT RP2 device
import time
from Core.Device import Device
from Core.Setting import DeviceSetting
from Config import get_config
# import slab   # use slab to generate stimulus

from traits.api import CFloat, Str, CInt, Enum, Any, Float, Bool
from Devices import TDTblackbox as tdt
import numpy as np
import os

import logging
log = logging.getLogger(__name__)


# NOTE: filter settings (Order and number of sets) are static, thus cannot be modified from Python
# TODO: ask about the filter delay. may need to modify the gate opening time in audio out device as well


class RX6RX8SpeakerCalSetting(DeviceSetting):
    """
    channel mapping
        uses digital output channels 0-3; channel 0 is used for camera trigger, and channels 1-3 are used to
        control up to 3 light sources. channel 0 will always output all the pulses, and channel 1-3 will be
        alternating when enabled
    pulse_width
        100 is full width, and 0 is none. pulse width does not affect channel 0
    """
    stim_length = Float(50., group='primary', dsec='length of stimulus, ms', reinit=False,
                        device='all', tag_name='stim_len_ms')
    stim_delay = Float(500, group='primary', dsec='onset delay of stimulus, ms', reinit=False,
                       device='all', tag_name='stim_delay_ms')
    use_pulse = Enum(0, 1, group='primary', dsec='if use the continuous stimulus',
                     reinit=False, tag_name='use_pulse', device='all')

    use_noise = Enum(0, 1, group='primary', dsec='if use the white noise stimulation',
                     reinit=False, tag_name='use_noise', device='RX8')
    use_tone = Enum(0, 1, group='primary', dsec='if use the pure tone stimulation',
                    reinit=False, tag_name='use_tone', device='RX8')
    use_custom = Enum(0, 1, group='primary', dsec='if use the custom audio from buffer component',
                      reinit=False, tag_name='use_custom', device='RX8')
    WN_amp = Float(1, group='primary', dsec='amplitude of white noise stimulus',
                   reinit=False, tag_name='wn_amp', device='RX8')
    tone_amp = Float(1, group='primary', dsec='amplitude of pure tone stimulus',
                     reinit=False, tag_name='tone_amp', device='RX8')
    tone_freq = Float(1000, group='primary', dsec='frequency of pure tone stimulus',
                      reinit=False, tag_name='tone_freq', device='RX8')
    stim_ch = CInt(17, group='primary', dsec='analog output channel to be used to drive the speaker',
                   reinit=False, tag_name='stim_ch', device='RX8')
    filter_on = Bool(False, group='primary', dsec='if use hardware filter; need to load the filter coefficients',
                     reinit=False, tag_name='filter_on', device='RX8')
    filter_select = CInt(1, group='primary', dsec='which filter set to be used',
                         reinit=False, tag_name='filter_sele', device='RX8')
    gate_delay_n = CInt(1, group='primary', dsec='gate delay w.r.t. stimulus buffer playing, needed when using hardware'
                                                 ' filter, n samples',
                        reinit=False, tag_name='gate_delay_n', device='RX8')

    # AI1 is channel 128 and AI2 is channel 129
    rec_ch = CInt(128, group='primary', dsec='analog input channel to be used to record the sound',
                  reinit=False, tag_name='rec_ch', device='RX6')
    sys_delay_n = CInt(1, group='primary', dsec='system delay from playing in RX8 to recording in RX6, n samples',
                       reinit=False, tag_name='sys_delay_n', device='RX6')

    device_freq = CFloat(97656.2500, group='status', dsec='sampling frequency of the device (Hz)')
    rcx_file_RX6 = Str('RCX\\RX6_speaker_calibration.rcx', group='status', dsec='the rcx file for RX6')
    rcx_file_RX8 = Str('RCX\\RX8_speaker_calibration.rcx', group='status', dsec='the rcx file for RX8')
    processor_RX6 = Str('RX6', group='status', dsec='name of the processor')
    processor_RX8 = Str('RX8', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    max_stim_length_n = CInt(500000, group='status', dsec='maximum length for stimulus in number of data points')
    gate_trf = Float(5., group='status', dsec='audio gating rise/fall time, ms')

    device_type = 'SpeakerCal_RX6RX8'


class RX6RX8SpeakerCal(Device):
    """
    the buffer 'PulseTTL' will not reset when calling pause/start. to reset the buffer, need to
    send software trigger 2 to the circuit, or use method reset_buffer
    """
    setting = RX6RX8SpeakerCalSetting()
    buffer = Any
    RX6 = Any
    RX8 = Any
    ZBus = Any
    _use_default_thread = False

    stimulus = Any
    filter_ntaps = CInt

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.RX6 = tdt.initialize_processor(processor=self.setting.processor_RX6,
                                            connection=self.setting.connection,
                                            index=1,
                                            path=os.path.join(expdir, self.setting.rcx_file_RX6))
        self.RX8 = tdt.initialize_processor(processor=self.setting.processor_RX8,
                                            connection=self.setting.connection,
                                            index=1,
                                            path=os.path.join(expdir, self.setting.rcx_file_RX8))
        # use ZBus in this case
        self.ZBus = tdt.initialize_zbus(connection=self.setting.connection)
        # not necessarily accurate
        TDT_freq = self.RX6.GetSFreq()  # not necessarily returns correct value
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
        for param in self._changed_params:
            par_trait = self.setting.trait(param)
            if par_trait.tag_name:
                # set the value of current trait into tag_name on TDT
                if par_trait.device == 'all':
                    tdt.set_variable(par_trait.tag_name, getattr(self.setting, param), self.RX6)
                    tdt.set_variable(par_trait.tag_name, getattr(self.setting, param), self.RX8)
                elif par_trait.device == 'RX6':
                    tdt.set_variable(par_trait.tag_name, getattr(self.setting, param), self.RX6)
                elif par_trait.device == 'RX8':
                    tdt.set_variable(par_trait.tag_name, getattr(self.setting, param), self.RX8)
                    # special case for filter_on
                    if param == 'filter_on':
                        tdt.set_variable('filter_off', not self.setting.filter_on, self.RX8)
                else:
                    raise ValueError('device {} is not recognized'.format(par_trait.device))
        # TODO: need to account for sound travel time?
        '''
        if 'stim_length' in self._changed_params:
            # set stimulus buffer length
            self.handle.SetTagVal('stimBufLen_n', self.setting.stim_length_n)
            # set recording buffer length
            self.handle.SetTagVal('recBufLen_n', self.setting.stim_length_n + 200)
        '''

    def _pause(self):
        # ZBus trigger: (racknum, trigtype, delay (ms)); 0 = all racks; 0 = pulse, 1 = high, 2 = low; minimum of 2
        self.ZBus.zBusTrigA(0, 2, 3)

    def _start(self):
        self.ZBus.zBusTrigA(0, 1, 3)

    def filter_enabled(self):
        """convenient method to check if the hardware filter is in use"""
        return self.setting.filter_on

    def get_filter_ntaps(self):
        """convenient method to get the number of taps for the hardware filter"""
        return self.filter_ntaps

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
            nsamples = self.RX6.GetTagVal('recBuf_idx')
        # the ReadTagV treats each 32-bit buffer as floats
        data = np.array(self.RX6.ReadTagV('audio_data', start_idx, nsamples), dtype=np.float32)
        # use np.ndarray.view to change the representation of dtype
        if np.issubdtype(dtype, np.integer):
            data = data.view(np.int32).astype(dtype)
        # data is 1 by n, should be uint8
        return data.reshape(-1, 1)

    def _stop(self):
        self.RX6.Halt()
        self.RX8.Halt()

    def load_stimulus(self, stimulus=None):
        """
        load stimulus into tdt buffer. stimulus should be prepared with slab toolbox
        Args:
            stimulus: generated from slab toolbox
        Returns:
            None
        """
        if stimulus:
            self.stimulus = stimulus
        if not self.stimulus:
            log.error('stimulus is not prepared. generate it using Slab toolbox')
        if self.stimulus.samplerate != self.setting.device_freq:
            log.error("stimulus sampling freq: {} is not the same as device sampling freq: {}".
                      format(self.stimulus.samplerate, self.setting.device_freq))
        data_length = self.stimulus.n_samples
        if self.stimulus.n_samples > self.setting.max_stim_length_n:
            log.warning("stimulus length: {} is longer than maximum length allowed: {}. stimulus is chopped.".
                        format(self.stimulus.n_samples, self.setting.max_stim_length_n))
            data_length = self.setting.max_stim_length_n
        self.RX8.WriteTagV('cust_stim', 0, self.stimulus.data[:data_length, 0])

    def load_filters(self, coefs, nTaps):
        """
        load filter coefficients into the hardware
        :param coefs: already shuffled and flattened filter coefficients. see TDT manual for more details
        :param nTaps: int, filter number of taps
        :return:
        """
        if len(coefs) % nTaps != 0:
            raise ValueError('length of coefficient array does not match the number of taps')
        self.filter_ntaps = int(nTaps)
        tdt.set_variable('filter_coefs', coefs, self.RX8)

    def is_playing(self):
        """
        :return: bool, if the stimulus is being played
        """
        return self.RX6.GetTagVal('is_running')

    def play_finished(self):
        """
        :return: bool, if the stimulus replay is finished, only makes sense when in 'pulse' mode
        """
        return self.RX6.GetTagVal('ran')

    def play_and_record(self):
        """
        play custom stimulus and return recorded result
        :return: (N-by-1) numpy array
        """
        started = False
        self.start()
        # need to detect the falling edge of the is_running tag
        while True:
            if not started:
                if self.play_finished():
                    started = True
            else:
                if not self.is_playing():
                    self.pause()
                    break
            time.sleep(0.05)
        return self.read()


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

    ld = RX6RX8SpeakerCal()
    # fd.setting.configure_traits()
