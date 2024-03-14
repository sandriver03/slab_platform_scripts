# light controller with TDT RP2 device

from Core.Device import Device
from Core.Setting import DeviceSetting
from Config import get_config

from traits.api import CFloat, Str, CInt, CArray, Any, Property, \
    TraitHandler, Trait, Dict, Instance
from tdt import DSPProject, DSPCircuit, DSPError
from tdt.convert import convert
import numpy as np
import os
import logging
log = logging.getLogger()


class validate_channels(TraitHandler):
    def validate(self, object, name, value):
        value = np.array([value], dtype=np.int8).ravel()
        if len(np.unique(value)) < len(value):
            raise ValueError('channels used must be unique!')
        if np.any(value < 0) or np.any(value > 3):
            raise ValueError('channels must be in range of 0-3')
        return value

    def info(self):
        return 'unique list of channels in range of 0-3'


class validate_width(TraitHandler):
    def validate(self, object, name, value):
        value = np.array([value], dtype=np.float).ravel()
        if np.any(value < 0) or np.any(value > 100):
            value[value < 0] = 0
            value[value > 100] = 100
            log.warning('width must be in range 0-100, coercing...')
        return value

    def info(self):
        return 'in range 0-100'


class LightControllerSetting(DeviceSetting):
    """
    channel mapping
        uses digital output channels 0-3; channel 0 is used for camera trigger, and channels 1-3 are used to
        control up to 3 light sources. channel 0 will always output all the pulses, and channel 1-3 will be
        alternating when enabled
    pulse_width
        100 is full width, and 0 is none. pulse width does not affect channel 0
    """

    pulse_freq = CFloat(50, group='primary', dsec='frequency of the pulse, Hz', reinit=False)
    pw_ch0 = Trait(10, validate_width(), group='primary',
                         dsec='width of the pulse on channel 0; 100=full width', reinit=False)
    pw_ch1 = Trait(100, validate_width(), group='primary',
                   dsec='width of the pulse on channel 1; 100=full width', reinit=False)
    pw_ch2 = Trait(100, validate_width(), group='primary',
                   dsec='width of the pulse on channel 2; 100=full width', reinit=False)
    pw_ch3 = Trait(100, validate_width(), group='primary',
                   dsec='width of the pulse on channel 3; 100=full width', reinit=False)
    channels_touse = Trait([0, 1, 2, 3], validate_channels(), group='primary',
                            dsec='which output channels to use', reinit=False)
    n_channels = CInt(4, group='status', dsec='total number of output channels, n')
    device_freq = CFloat(24414.0625, group='status', dsec='sampling frequency of the device (Hz)')
    rcx_file = Str('RCX\\lightRP2.rcx', group='status', dsec='name of the rcx file to load')
    processor = Str('RP2', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    device_type = 'LightController_RP2'

    # intermediate variables
    pulse_period_n = Property(CInt, depends_on=['pulse_freq', 'device_freq'],
                              group='derived', dsec='period of the pulse, n samples')
    chs_delay_n = Property(CArray, depends_on=['channels_touse', 'pulse_freq', 'device_freq'],
                         group='derived', dsec='delays on channel 0-3, n samples')
    chs_wd_n = Property(CArray, depends_on=['channels_touse', 'pulse_freq', 'pwd_all', 'device_freq'],
                      group='derived', dsec='width of each pulse on channels in use, n samples')
    pwd_all = Property(CArray, depends_on=['pw_ch0', 'pw_ch1', 'pw_ch2', 'pw_ch3'],
                       group='status', dsec='width of each pulse on channel 0-3, 0-100 percentage')

    def _get_pulse_period_n(self):
        return convert('ms', 'n', 1/self.pulse_freq*1000, self.device_freq)

    def _get_pwd_all(self):
        return np.array([self.pw_ch0, self.pw_ch1, self.pw_ch2, self.pw_ch3]).ravel()

    def _get_chs_delay_n(self):
        delays = np.zeros(len(self.channels_touse), dtype=int)
        if 0 in self.channels_touse:
            delays[~np.equal(self.channels_touse, 0)] = \
                np.linspace(0, self.pulse_period_n, len(self.channels_touse))[:-1]
        else:
            delays = np.linspace(0, self.pulse_period_n, len(self.channels_touse)+1)[:-1]
        # delay of 0 will generate no pulse; so for channels in use, the minimum should be set to 1
        delays[delays == 0] = 1
        delays_all = np.zeros(self.n_channels, dtype=int)
        delays_all[self.channels_touse] = delays
        return delays_all

    def _get_chs_wd_n(self):
        n_ch = np.sum(~np.equal(self.channels_touse, 0))
        wd = np.int32(self.pulse_period_n/n_ch*self.pwd_all[self.channels_touse]/100)
        # make sure the width + delay is smaller than pulse period
        wd[self.pwd_all[self.channels_touse] == 100] = wd[self.pwd_all[self.channels_touse] == 100] - 2
        return wd


class LightController(Device):
    """
    the buffer 'PulseTTL' will not reset when calling pause/start. to reset the buffer, need to
    send software trigger 2 to the circuit, or use method reset_buffer
    """
    setting = LightControllerSetting()
    circuit = Instance(DSPCircuit)
    project = Instance(DSPProject)
    buffer = Any
    _use_default_thread = False

    ParaTag_mapping = Dict({'pulse_period': 'PulsePeriod',
                'chs_delay': ['Ch0del', 'Ch1del', 'Ch2del', 'Ch3del'],
                'chs_wd': ['Ch0wd', 'Ch1wd', 'Ch2wd', 'Ch3wd']})

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.project = DSPProject(interface=self.setting.connection)
        self.circuit = self.project.load_circuit(os.path.join(expdir, self.setting.rcx_file),
                                                 self.setting.processor)
        self.setting.device_freq = self.circuit.fs
        self.buffer = self.circuit.get_buffer('PulseTTL', 'r', src_type='int', dest_type='uint8')
        self.circuit.start()
        self._output_specs = {}

    def _configure(self, **kwargs):
        # configure pulse period
        self.circuit.set_tag('PulsePeriod', self.setting.pulse_period_n)
        # configure pulse delay and pulse width; pulse delay need to be configured to all channels
        for idx, vd in enumerate(self.setting.chs_delay_n):
            self.circuit.set_tag(self.ParaTag_mapping['chs_delay'][idx], vd)
        for idx, ch in enumerate(self.setting.channels_touse):
            self.circuit.set_tag(self.ParaTag_mapping['chs_wd'][ch], self.setting.chs_wd_n[idx])

    def _pause(self):
        self.circuit.trigger(1)

    def _start(self):
        self.circuit.trigger(1)

    def reset_buffer(self):
        self.circuit.trigger(2)
        self.buffer.read_index = 0

    def read(self, nsamples=None):
        """
        Read TTL signal from the PulseTTL buffer
        the TTL signal is read as 8 bit unsigned integers. unpack it into 8 bit to get 8 channels
        Only 4 channels are used currently.
        Args:
            nsamples: number of samples to read. if None, read all available samples

        Returns:
            8-by-N array, TTL from 8 channels
        """
        return self.buffer.read(nsamples)   # data is 1 by n, should be uint8
        # return np.unpackbits(data, 0)

    def _stop(self):
        self.circuit.load()  # this will reset every tag to default, and stop the circuit


if __name__ == '__main__':
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

    ld = LightController()
    # fd.setting.configure_traits()