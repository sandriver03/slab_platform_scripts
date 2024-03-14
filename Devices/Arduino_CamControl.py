from .Arduino import ArduinoDevice, ArduinoSetting

from traits.api import Instance, Int


class ArduinoCamCtrlSetting(ArduinoSetting):

    output_channels = Int(1, group='primary', dsec='pattern of light output channels; '
                                                   'the binary representation is the pattern',
                          reinit=False)
    interleave_channels = Int(0, group='primary', dsec='if all output channels are used together or not',
                              reinit=False)

    Arduino_name = 'Mega-Cam'


class ArduinoCamCtrl(ArduinoDevice):

    setting = Instance(ArduinoCamCtrlSetting, ())

    _use_default_thread = False

    def _start(self):
        self._sendMessage('c', [self.setting.output_channels, self.setting.interleave_channels])

    def _pause(self):
        self._sendMessage('i')

    def _stop(self):
        self.serialPort.close()
