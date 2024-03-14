from Devices.Cam_PhotometricsPrime import PrimeCam
from Devices.TDT_RX6_ImagingExp import RX6ImagingController
from Core.Setting import ExperimentSetting
from Core.ExperimentLogic import ExperimentLogic
from Core.Data import ExperimentData
from Ultilities.arraytools import downscale_nparray
# from Ultilities import Speaker

from traits.api import List, CInt, Float, Property, cached_property, Enum, Int
import numpy as np
import random
import threading
import logging
log = logging.getLogger(__name__)


class Imaging_Tonotopy_Setting(ExperimentSetting):
    experiment_name = 'ImagingExperiment'

    stim_type = Enum('Tone', 'GNoise', group='primary', dsec='type of stimulus to be used', reinit=False,)
    tone_freqs = List([4000, 8000, 16000], group='primary',
                      dsec='list of different tone frequency to be used, Hz', reinit=False)
    tone_amps = List([3.55, ], group='primary', dsec='tone amplitudes to be used in each frequency', reinit=False)
    GNoise_amps = List([3.55, ], group='primary', dsec='gaussian noise amplitudes to be used', reinit=False)
    modulation_depth = List([0, ], group='primary',
                            dsec='tone modulation depth to be used in each frequency', reinit=False)
    modulation_freq = List([20, ], group='primary',
                           dsec='tone modulation frequency to be used in each frequency', reinit=False)
    trial_duration = 10
    trial_number = 20
    stimOn_delay = Float(3, group='primary', dsec='stimulus onset delay, s', reinit=False)
    stim_len = Float(0.1, group='primary', dsec='stimulus length, s', reinit=False)
    n_stim = Int(5,  group='primary', dsec='number of stimuli to be generated', reinit=False)
    stim_interv = Float(0.15,  group='primary', dsec='time interval between consecutive stimuli, s', reinit=False)

    device_freq = Float(97656.2500, group='status', dsec='sampling frequency of the device (Hz)')

    stimOn_delay_n = Property(depends_on=['stimOn_delay', 'device_freq'], group='derived',
                              dsec='stimulus onset delay, number of samples')
    stim_len_n = Property(depends_on=['stim_len', 'device_freq'], group='derived',
                          dsec='stimulus length, number of samples')
    trial_len_n = Property(depends_on=['trial_duration', 'device_freq'], group='derived',
                           dsec='trial length, number of samples')
    total_trial = Property(CInt(), group='status', depends_on=['trial_number', 'tone_freqs', 'tone_amps'],
                           dsec='Total number of trials')
    stim_interv_n = Property(depends_on=['stim_interv', 'device_freq'], group='derived',
                             dsec='time interval between consecutive stimuli, number of samples')

    def _get_total_trial(self):
        if self.stim_type == 'Tone':
            return int(self.trial_number * self.tone_freqs.__len__() * self.tone_amps.__len__())
        elif self.stim_type == 'GNoise':
            return int(self.trial_number * self.GNoise_amps.__len__())
        else:
            raise ValueError('stimulus type {} is not implemented'.format(self.stim_type))

    @cached_property
    def _get_stimOn_delay_n(self):
        return int(self.stimOn_delay * self.device_freq)

    @cached_property
    def _get_stim_len_n(self):
        return int(self.stim_len * self.device_freq)

    @cached_property
    def _get_trial_len_n(self):
        return int(self.trial_duration * self.device_freq)

    @cached_property
    def _get_stim_interv_n(self):
        return int(self.stim_interv * self.device_freq)


class Imaging_Tonotopy(ExperimentLogic):
    setting = Imaging_Tonotopy_Setting()
    data = ExperimentData()
    time_0 = Float()

    @staticmethod
    def _calculate_modulation_params(amp, mod_depth):
        """
        calculate tone modulation parameters
        Args:
            amp: float, output amplitude in V
            mod_depth: float, modulation depth
        Returns:
            tuple of (modulator_amp, modulator_shift)
        """
        # modulator amplitude and shift should be tone_amp = m_amp + m_shift
        #                                         m_depth  = m_amp / (m_amp + m_shift)
        m_amp = amp * mod_depth
        m_shift = amp - m_amp
        return m_amp, m_shift

    def _devices_default(self):
        fd = PrimeCam()
        ld = RX6ImagingController()
        return {'Cam': fd, 'RX6': ld}

    def _initialize(self, **kwargs):
        # configure camera trigger mode to edge, exposure out mode to all rows, and clear mode to pre sequence
        # we dont really need camera buffer
        # let's use 2x2 binning for now
        self.devices['Cam'].configure(clear_mode='preExp',
                                      trigger_mode='Edge',
                                      exp_out_mode='All Rows',
                                      buffer_time=0,
                                      binning='2x2',
                                      exposure_time=4)
        # configure light pulse parameters
        self.devices['RX6'].configure(pl_ch0=1.8, pl_ch1=3.9, channels_touse=[1])
        # turn off camera fan
        self.devices['Cam'].set_fan_speed('off')
        # use frame rate of 21 for 1x1 binning, or 30 for 2x2 binning
        self.devices['RX6'].configure(pulse_freq=30)

    # internal temporal parameter
    tone_list = List()
    modulator_params = List()

    def _configure(self, **kargs):
        # configure trial duration, stimulus onset and stimulus length in the TDT device
        if 'trial_duration' in self._changed_params:
            self.devices['RX6'].handle.SetTagVal('trial_dur_n', self.setting.trial_len_n)
        if 'stimOn_delay' in self._changed_params:
            self.devices['RX6'].handle.SetTagVal('stim_delay_n', self.setting.stimOn_delay_n)
        if 'stim_len' in self._changed_params:
            self.devices['RX6'].handle.SetTagVal('stim_len_n', self.setting.stim_len_n)
        if 'n_stim' in self._changed_params:
            self.devices['RX6'].handle.SetTagVal('n_stim', self.setting.n_stim)
        if 'stim_interv' in self._changed_params:
            self.devices['RX6'].handle.SetTagVal('stim_interv_n', self.setting.stim_interv_n)

    def setup_experiment(self, info=None):
        if not self.tone_list:
            if self.setting.stim_type == 'Tone':
                self.tone_list, self.modulator_params = self._generate_tone_sequence()
                # set tag in TDT device
                self.devices['RX6'].handle.SetTagVal('use_tone', 1)
                self.devices['RX6'].handle.SetTagVal('use_noise', 0)
            elif self.setting.stim_type == 'GNoise':
                self.tone_list, self.modulator_params = self._generate_noise_sequence()
                # set tag in TDT device
                self.devices['RX6'].handle.SetTagVal('use_tone', 0)
                self.devices['RX6'].handle.SetTagVal('use_noise', 1)
            else:
                raise ValueError('stimulus type {} not known'.format(self.setting.stim_type))
        # save the sequence
        self._tosave_para['tone_sequence'] = self.tone_list
        # configure trial duration, stimulus onset and stimulus length in the TDT device
        # self.devices['RX6'].handle.SetTagVal('trial_dur_n', self.setting.trial_len_n)
        # self.devices['RX6'].handle.SetTagVal('stim_delay_n', self.setting.stimOn_delay_n)
        # self.devices['RX6'].handle.SetTagVal('stim_len_n', self.setting.stim_len_n)

        # save snapshot images
        N_snapshot = self.devices['Cam'].N_snapshot
        if N_snapshot == 0:
            log.warning('no snapshot has been taken!')
        else:
            self._tosave_para['snapshot'] = self.devices['Cam'].buffer_snapshot[:N_snapshot]
        # set data class to use thread writer
        self.data.writer_type = 'subprocess'
        # put stream handler in writer_params
        self.data.configure_writer(stream_handler={'PrimeCam_0_PrimeCam_0_tcp': cam_stream_handler})

    def _generate_tone_sequence(self):
        seq = []
        for f in self.setting.tone_freqs:
            for a in self.setting.tone_amps:
                for md in self.setting.modulation_depth:
                    for mf in self.setting.modulation_freq:
                        seq.extend(((f, a, mf, md), ) * self.setting.trial_number)
        random.shuffle(seq)
        # calculate modulator params
        mod_params = []
        for par in seq:
            mod_params.append(self._calculate_modulation_params(par[1], par[3]))
        return seq, mod_params

    def _generate_noise_sequence(self):
        seq = []
        for a in self.setting.GNoise_amps:
            for md in self.setting.modulation_depth:
                for mf in self.setting.modulation_freq:
                    seq.extend(((a, mf, md), ) * self.setting.trial_number)
        random.shuffle(seq)
        # calculate modulator params
        mod_params = []
        for par in seq:
            mod_params.append(self._calculate_modulation_params(par[0], par[2]))
        return seq, mod_params

    def _before_start(self):
        # self.data.close_input('ShamCam_0_ShamCam_0_tcp')
        # self.data.writer.worker.register_stream_handler('PrimeCam_0_PrimeCam_0_tcp', cam_stream_handler)
        # start camera; it should be already configured
        if self.devices['Cam'].state == 'Running':
            self.devices['Cam'].reset_frameInfo()
        elif self.devices['Cam'].state in ('Paused', 'Ready'):
            self.devices['Cam'].start()
        else:
            raise RuntimeError('Camera cannot be operated!')

    def _start_trial(self):
        # configure TDT device to setup stimulus parameters
        tone_param = self.tone_list[self.setting.current_trial]
        mod_param = self.modulator_params[self.setting.current_trial]
        if self.setting.stim_type == 'Tone':
            self.devices['RX6'].handle.SetTagVal('tone_freq', tone_param[0])
            self.devices['RX6'].handle.SetTagVal('mod_amp', mod_param[0])
            self.devices['RX6'].handle.SetTagVal('mod_shift', mod_param[1])
            self.devices['RX6'].handle.SetTagVal('mod_freq', tone_param[2])
        elif self.setting.stim_type == 'GNoise':
            self.devices['RX6'].handle.SetTagVal('mod_amp', mod_param[0])
            self.devices['RX6'].handle.SetTagVal('mod_shift', mod_param[1])
            self.devices['RX6'].handle.SetTagVal('mod_freq', tone_param[1])
        else:
            raise ValueError('stimulus type {} not known'.format(self.setting.stim_type))
        # start TDT device
        self.devices['RX6'].start()
        # use trial timer to time and end trial
        self.trial_timer = threading.Timer(self.setting.trial_duration, self.trial_stop_fired)
        self.trial_timer.start()

    def _stop_trial(self):
        self.devices['RX6'].pause()
        # read timing info from TDT device and save it
        RX6_timing_sig = self.devices['RX6'].read()
        self.data.write('ImagingExp_RX6_0', RX6_timing_sig)
        # put a stop frame into camera stream
        self.devices['Cam'].generate_stop_frame()
        # save and reset camera timing info
        self.data.write('PrimeCam_0_timing', np.array(self.devices['Cam']._frameTms))
        self.devices['Cam'].reset_frameInfo()
        # set trial end flags
        self.data.input_finished(('event_log', 'trial_log', 'ImagingExp_RX6_0', 'PrimeCam_0_timing'))
        # save data
        self.data.save(close_file=False)

    def _stop(self):
        pass

    def _pause(self):
        try:
            self.devices['RX6'].pause()
        except AssertionError:
            pass


def cam_stream_handler(obj, stream, stop_sig=b's', data_sig=b'd', preview_sig=b'p'):
    """
    read camera stream data
    Args:
        obj: the writer.worker instance
        stream: an Stream.InputStream object
        stop_sig: byte, stop signal from the stream
        data_sig: byte, stream packets need to be saved
        preview_sig: byte, stream packets need to be skipped
    Returns:
    """
    # first check how many packets are queued up
    try:
        packet_to_read = stream.monitor()[0] - stream.N_packet
    except TypeError:  # stream.monitor returns None because no new data arrived
        if stream.latest_packet_N is not None:
            packet_to_read = stream.latest_packet_N - stream.N_packet
        else:
            packet_to_read = stream.Idx_initial_packet - stream.N_packet
    if packet_to_read <= 0:
        return
    flag_idx = obj.input_names.index(stream.name)
    if not obj._input_finished_flags[flag_idx]:
        # create array
        data = np.zeros(((packet_to_read,) + stream.params['shape'][1:]),
                        dtype=stream.params['dtype'])
        # read queued packets and save it to file
        # only do this when there is actual data to save, i.e. camera is not in preview mode
        tosave_idx = 0
        for idx in range(packet_to_read):
            s_data = stream.recv()
            # preview images tagged with b'p' can be ignored
            if s_data[2][1] == stop_sig:
                data = data[:tosave_idx]
                obj.input_finished(obj._current_trial, (stream.name,))
                break
            elif s_data[2][1] == data_sig:
                data[idx] = s_data[1]
                tosave_idx += 1

        if tosave_idx > 0:
            # down scale array if software scaling is specified
            if 'software_scaling' in stream.params and stream.params['software_scaling']:
                data = downscale_nparray(data, (1, ) + stream.params['sf_full'])
            obj.write(stream.name, data, obj._current_trial)


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

    sl = load_cohort('example')
    sl.read_from_h5file()

    fe = Imaging_Tonotopy(subject=sl.subjects[0])
    fd = fe.devices['Cam']
    ld = fe.devices['RX6']
    # fe.start()
