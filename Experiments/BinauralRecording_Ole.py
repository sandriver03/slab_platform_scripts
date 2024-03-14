from Core.Setting import ExperimentSetting
from Core.ExperimentLogic import ExperimentLogic
from Core.Data import ExperimentData
from Devices.TDT_RX8_Ole import RX8_Ole_Device
from traits.api import List

import random


class BinauralRecordingSetting(ExperimentSetting):
    speaker_list = List(group='primary', dsec='')


class BinauralRecordingExperiment(ExperimentLogic):

    setting = BinauralRecordingSetting()
    data = ExperimentData()
    sequence = List()

    def _initialize(self, **kargs):
        self.devices['RX8'] = RX8_Ole_Device()
        self.devices['RX8'].experiment = self

    def setup_experiment(self, info=None):
        self.sequence = random.shuffle(self.setting.speaker_list)

    def generate_stimulus(self):
        pass

    def _prepare_trial(self):
        self.devices['RX8'].configure(channel_nr=self.sequence[0])
        self.devices['RX8'].configure(stimulus=self.generate_stimulus())

    def _start_trial(self):
        self.devices['RX8'].start()
