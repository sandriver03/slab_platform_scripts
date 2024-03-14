'''
    a testing case, operation on the FooDevice
'''

from traits.api import List, Instance, Float, Property, Int
import random
import numpy as np
import time

from Config import get_config
from Core.ExperimentLogic import ExperimentLogic
from Core.Setting import ExperimentSetting
from Devices.FooDevice import FooDevice
from Core.Data import ExperimentData
import logging


class FooExperimentSetting(ExperimentSetting):
    experiment_name = 'FooExp'
    mu_sequence = List([1, 1], group='primary', context=False, dsec='different means of the Gaussian to run')
    trial_duration = 5
    trial_number = 5
    test_large_val = Instance(np.ndarray, group='primary', context=False, dsec='test large val')

    total_trial = Property(Int(), group='status', depends_on=['trial_number'],
                                    dsec='Total number of trials')

    def _test_large_val_default(self):
        return np.random.randn(2, 1)


    def _get_total_trial(self):
        return self.trial_number


class FooExperiment(ExperimentLogic):
    setting = FooExperimentSetting()
    data = ExperimentData()
    time_0 = Float()

    def _devices_default(self):
        fd = FooDevice()
        fd.setting.device_ID = 0
        return {'FooDevice': fd}

    def _initialize(self, **kwargs):
        pass

    # internal temporal parameter
    mu_list = List()

    def setup_experiment(self, info=None):
        self.mu_list = []
        for k in self.setting.mu_sequence:
            self.mu_list.extend([k]*self.setting.trial_number)
        # randomize the sequence
        random.shuffle(self.mu_list)
        # save the sequence
        self._tosave_para['mu_sequence'] = self.mu_list

        # setup correct data_length in FooDevice
        data_length = self.devices['FooDevice'].setting.sampling_freq * self.setting.trial_duration
        self.devices['FooDevice'].configure(data_length=data_length)
        self.time_0 = time.time()

    def _start_trial(self):
        self.devices['FooDevice'].configure(mu=self.mu_list[self.setting.current_trial])
        self.devices['FooDevice'].start()
        log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop_trial(self):
        # read data from FooDevice
        self.data.write('FooDevice_0', self.devices['FooDevice'].buffer)
        # self.data.current['FooDevice_0'].write(self.devices['FooDevice'].buffer)
        # save data
        self.data.save()
        log.info('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop(self):
        pass

    def _pause(self):
        pass


if __name__ == '__main__':
    from Core.subject import load_cohort
    import logging

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

    cohort = load_cohort('example')

    fe = FooExperiment(subject=cohort.subjects[1])
    fd = fe.devices['FooDevice']
    # fe.start()
