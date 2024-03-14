from Config import get_config
from Core.Device import Device
from Core.Setting import DeviceSetting
from traits.api import CFloat, CInt, Str, Any, Instance
import threading

from Devices import TDTblackbox as tdt
import logging

log = logging.getLogger(__name__)

class RX8_Ole_Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    rx8_file       = Str('RCX\\play_mono.rcx', group='primary', dsec='name of the rcx file to load')
    processor      = Str('RX8', group='status', dsec='name of the processor')
    connection     = Str('GB', group='status', dsec='')
    index          = CInt(1, group='primary', dsec='index of the device to connect to')
    stimulus       = Any(group='primary', dsec='stimulus to play', context=False)
    channel_nr     = CInt(1, group='primary', dsec='channel to play sound', context=False)


class RX8_Ole_Device(Device):
    setting = RX8_Ole_Setting()
    handle = Any
    thread = Instance(threading.Thread)

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor="RX8", connection="GB", index=1, path=expdir + "rpvdsx/play_mono.rcx")
        self.handle.Run()

        # create thread to monitoring hardware
        if not self.thread or not self.thread.isAlive():
            log.debug('creating thread...')
            self.thread = threading.Thread(target=self.thread_func, daemon=True)
            self.thread.start()

    def _configure(self, **kargs):
        if self.stimulus.__len__():
            self.handle.WriteTagV('datain', 0, self.stimulus)
            self.handle.SetTagVal('playbuflen', len(self.stimulus))

        self.handle.SetTagVal('channelnr', self.channel_nr)
        log.debug('output channel change to {}'.format(self.channel_nr))

    def _start(self):
        self.handle.SoftTrg(1)

    def _terminate(self):
        self.handle.Halt()

    def thread_func(self):
        while self.handle.GetTagVal('playback'):
            pass
        self.stop()
        self.experiment._trial_stop = True
