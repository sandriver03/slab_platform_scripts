"""
use the RX6-RX8 to perform different imaging experiments:
    1. experiments with single stimulus per trial
    2. experiments with rolling stimuli in one trial

"""
# add a testing mode, which runs without the camera (use a simulated camera)
# NOTE: enable filters when doing spatial experiments, but disable them when not

from Devices.Cam_PhotometricsPrime import PrimeCam
from Devices.emptyCam import EmptyCam
from Devices.TDT_RX6RX8_Imaging import RX6RX8ImagingController
from Core.Setting import ExperimentSetting
from Core.ExperimentLogic import ExperimentLogic
from Core.Data import ExperimentData
from Ultilities.arraytools import downscale_nparray
from Speakers.speaker_config import SpeakerArray
from Speakers.utilities import prepare_hwfilter_coefs, dB_to_Vrms

from traits.api import List, CInt, Float, Property, cached_property, Enum, Str, Any, Bool
from collections import Iterable
import numpy as np
import random
import threading
import logging
log = logging.getLogger(__name__)


# default settings in the TDT controller for different experiment modes
settings_single = {'exp_params': {'ISI': 10},
                   'TDT_params': {'n_burst': 5, 'ISI_burst': 0.15, 'burst_len': 0.1,
                                  'use_adaptor': 0, 'adap_ch': 0}}
settings_rolling = {'exp_params': {'ISI': 3},
                    'TDT_params': {'n_burst': 1, 'ISI_burst': 1.5, 'burst_len': 0.7,
                                   'use_adaptor': 0, 'adap_ch': 0}}
settings_single_adaptor = {'exp_params': {'ISI': 10},
                           'TDT_params': {'n_burst': 1, 'ISI_burst': 1.5, 'burst_len': 0.1, 'prob_delay': 0.6,
                                          'use_adaptor': 1, 'adap_delay': 0., 'adap_len': 0.6, 'adap_ch': 16}}
settings_rolling_adaptor = {'exp_params': {'ISI': 3},
                            'TDT_params': {'n_burst': 1, 'ISI_burst': 1.5, 'burst_len': 0.1, 'prob_delay': 0.6,
                                           'use_adaptor': 1, 'adap_delay': 0., 'adap_len': 0.6, 'adap_ch': 16}}
default_mode_params = {'Single': settings_single,
                       'Single_adaptor': settings_single_adaptor,
                       'Rolling': settings_rolling,
                       'Rolling_adaptor': settings_rolling_adaptor}

# TODO: generate spatial stimuli based on speaker array


class ImagingExps_Setting(ExperimentSetting):
    experiment_name = 'ImagingExperiment'

    trial_number = 20

    mode = Enum('Single', 'Single_adaptor', 'Rolling', 'Rolling_adaptor',
                group='primary', dsec='if multiple stimuli in single trials', reinit=False,)
    stim_type = Enum('GNoise', 'Tone', group='primary', dsec='type of stimulus to be used', reinit=False,)
    tone_freqs = List([4000, 8000, 16000], group='primary',
                      dsec='list of different tone frequency to be used, Hz', reinit=False)
    tone_amps = List([75, ], group='primary', dsec='tone amplitudes to be used in each frequency, dB', reinit=False)
    GNoise_amps = List([75, ], group='primary', dsec='gaussian noise amplitudes to be used, dB', reinit=False)
    stim_locs = List([16, ], group='primary', dsec='stimulus locations', reinit=False)
    # Nstim_per_trial = CInt(group='primary', dsec='number of stimuli in each trial', reinit=False)

    # TDT parameters, but needed for experiment control, so should be set from Experiment class
    ISI = Float(3, group='primary', dsec='inter-stimulus interval, s', reinit=False, TDT_param=True)
    stimOn_delay = Float(3, group='primary', dsec='stimulus onset delay, s', reinit=False, TDT_param=True)
    stim_repeat_per_trial = CInt(10, group='primary', reinit=False,
                                 dsec='number of repeats for the stimuli set in rolling experiment')

    # speaker configurations
    # TODO: currently do not allow to change those
    speaker_setup = Str("IMAGING", group='status',
                        dsec='speaker configuration name')

    N_stims = Property(group='status', dsec='number of stimuli per trial',
                       depends_on=['mode', 'stim_repeat_per_trial', 'stim_type',
                                   'tone_freqs', 'tone_amps', 'GNoise_amps', 'stim_locs'])
    total_trial = Property(CInt, group='status', dsec='Total number of trials',
                           depends_on=['mode', 'trial_number', 'stim_type',
                                       'tone_freqs', 'tone_amps', 'GNoise_amps', 'stim_locs'])
    trial_duration = Property(group='status', dsec='length of each trial, s',
                              depends_on=['mode', 'stim_repeat_per_trial', 'ISI', 'stim_type',
                                          'tone_freqs', 'tone_amps', 'GNoise_amps', 'stim_locs'])

    @cached_property
    def _get_N_stims(self):
        if self.mode in ('Single', 'Single_adaptor'):
            return int(1)
        elif self.mode in ('Rolling', 'Rolling_adaptor'):
            if self.stim_type == 'Tone':
                n_stim_per_trial = self.stim_repeat_per_trial * len(self.tone_freqs) * \
                                   len(self.tone_amps) * len(self.stim_locs)
                return int(n_stim_per_trial)
            elif self.stim_type == 'GNoise':
                n_stim_per_trial = self.stim_repeat_per_trial * len(self.GNoise_amps) * len(self.stim_locs)
                return int(n_stim_per_trial)
            else:
                raise ValueError('stimulus type {} is not implemented'.format(self.stim_type))
        else:
            raise ValueError('experiment mode {} is not implemented'.format(self.mode))

    @cached_property
    def _get_total_trial(self):
        if self.mode in ('Single', 'Single_adaptor'):
            if self.stim_type == 'Tone':
                return int(self.trial_number * self.tone_freqs.__len__() * self.tone_amps.__len__()
                           * self.stim_locs.__len__())
            elif self.stim_type == 'GNoise':
                return int(self.trial_number * self.GNoise_amps.__len__() * self.stim_locs.__len__())
            else:
                raise ValueError('stimulus type {} is not implemented'.format(self.stim_type))
        elif self.mode in ('Rolling', 'Rolling_adaptor'):
            return int(self.trial_number)
        else:
            raise ValueError('experiment mode {} is not implemented'.format(self.mode))

    @cached_property
    def _get_trial_duration(self):
        if self.mode in ('Single', 'Single_adaptor'):
            return self.ISI
        elif self.mode in ('Rolling', 'Rolling_adaptor'):
            if self.stim_type == 'Tone':
                n_stim_per_trial = self.stim_repeat_per_trial * self.tone_freqs.__len__() * self.tone_amps.__len__() \
                                   * len(self.stim_locs)
                return self.ISI * n_stim_per_trial + self.stimOn_delay + 7
            elif self.stim_type == 'GNoise':
                n_stim_per_trial = self.stim_repeat_per_trial * self.GNoise_amps.__len__() * len(self.stim_locs)
                return self.ISI * n_stim_per_trial + self.stimOn_delay + 7
            else:
                raise ValueError('stimulus type {} is not implemented'.format(self.stim_type))
        else:
            raise ValueError('experiment mode {} is not implemented'.format(self.mode))


class ImagingExps(ExperimentLogic):
    setting = ImagingExps_Setting()
    data = ExperimentData()
    time_0 = Float()
    _testing_mode = False
    _camera_name = Str
    speakers = Any
    calib_const = Float

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
        if self._testing_mode:
            fd = EmptyCam()
            self._camera_name = 'EmptyCam_0'
        else:
            fd = PrimeCam()
            self._camera_name = 'PrimeCam_0'
        ld = RX6RX8ImagingController()
        return {'Cam': fd, 'TDTs': ld}

    def _initialize(self, **kwargs):
        # configure camera trigger mode to edge, exposure out mode to all rows, and clear mode to pre sequence
        # we dont really need camera buffer
        # let's use 2x2 binning for now
        self.devices['Cam'].configure(clear_mode='preExp',
                                      trigger_mode='Edge',
                                      exp_out_mode='All Rows',
                                      buffer_time=0,
                                      binning='2x2',
                                      exposure_time=10)
        # configure light pulse parameters
        self.devices['TDTs'].configure(pl_ch0=2.4, pl_ch1=9.9, pl_ch2=9.9, channels_touse=[1])
        # turn off camera fan
        self.devices['Cam'].set_fan_speed('off')
        # use frame rate of 21 for 1x1 binning, or 30 for 2x2 binning
        self.devices['TDTs'].configure(pulse_freq=30)
        # depending on the mode, set defaults for each mode
        self.configure(**default_mode_params[self.setting.mode]['exp_params'])
        self.configure(TDTs=default_mode_params[self.setting.mode]['TDT_params'])
        # speaker settings
        self.speakers = SpeakerArray(setup=self.setting.speaker_setup)
        self.speakers.load_speaker_table()
        # TODO: currently only load latest calibration; others need to be manually loaded
        self.speakers.load_calibration()
        self.calib_const = self.speakers.calib_result['SPL_const']
        # load filter coefficients
        # need to load 2 sets of filters, one for adaptor and one for probe; they are the same in this case
        coefs, ntaps = prepare_hwfilter_coefs(self.speakers.speakers)
        self.devices['TDTs'].load_filters(coefs, ntaps, tag='adap_filter_coefs')
        self.devices['TDTs'].load_filters(coefs, ntaps, tag='prob_filter_coefs')
        # turn on filter
        self.devices['TDTs'].configure(filter_on=True)

    def filter_enabled(self):
        """convenient method to check if the hardware filter is in use"""
        return self.devices['TDTs'].filter_enabled()

    # internal temporal parameter
    tone_list = List()
    tone_name_list = List()
    modulator_params = List()

    def _configure(self, **kargs):
        # configure ISI and stimulus onset in the TDT device
        TDT_params = {}
        if 'ISI' in self._changed_params:
            TDT_params['ISI'] = self.setting.ISI
        if 'stimOn_delay' in self._changed_params:
            TDT_params['stimOn_delay'] = self.setting.stimOn_delay
        if TDT_params:
            self.configure(TDTs=TDT_params)

    def setup_experiment(self, info=None):
        if not self.tone_list:
            if self.setting.stim_type == 'Tone':
                self.tone_list, self.tone_name_list = self._generate_tone_sequence()
            elif self.setting.stim_type == 'GNoise':
                self.tone_list, self.tone_name_list = self._generate_noise_sequence()
            else:
                raise ValueError('stimulus type {} not known'.format(self.setting.stim_type))
        # set tag in TDT device
        if self.setting.stim_type == 'Tone':
            self.devices['TDTs'].configure(use_tone=1, use_noise=0)
        elif self.setting.stim_type == 'GNoise':
            self.devices['TDTs'].configure(use_tone=0, use_noise=1)
        else:
            raise ValueError('stimulus type {} not known'.format(self.setting.stim_type))
        # save the sequence
        self._tosave_para['tone_sequence'] = self.tone_list
        self._tosave_para['stim_name_sequence'] = self.tone_name_list

        # save snapshot images
        N_snapshot = self.devices['Cam'].N_snapshot
        if N_snapshot == 0:
            log.warning('no snapshot has been taken!')
        else:
            self._tosave_para['snapshot'] = self.devices['Cam'].buffer_snapshot[:N_snapshot]
        # set data class to use thread writer
        self.data.writer_type = 'subprocess'
        # put stream handler in writer_params
        self.data.configure_writer(stream_handler={'{}_{}_tcp'.format(self._camera_name, self._camera_name):
                                                   cam_stream_handler})

    # TODO: use Marc's trialsmaker code to generate non-repeating sequences
    # TODO: check the adaptor sequence
    # need channel_analog to map to TDT output, and id for filter select
    def _generate_tone_sequence(self):
        """
        sequence need to be a list of lists, [trials][stims in each trial]
        :return:
        """
        seq = []
        seq_names = []
        amps_in_Vrms = dB_to_Vrms(self.setting.tone_amps, self.calib_const)
        filter_idx = self._stimloc_to_filteridx(self.setting.stim_locs)
        if self.setting.mode in ('Single', 'Single_adaptor'):
            for freq in self.setting.tone_freqs:
                for amp, dB in zip(amps_in_Vrms, self.setting.GNoise_amps):
                    for loc, f_idx in zip(self.setting.stim_locs, filter_idx):
                        seq.extend((([freq], [amp], [loc], [f_idx]), ) * self.setting.trial_number)
                        seq_names.extend((([freq], [dB], [loc]), ) * self.setting.trial_number)
            # convert names sequence into np array
            seq_names = np.array(seq_names, dtype=np.int32).squeeze()
            # shuffling
            s_idx = np.random.permutation(len(seq))
            seq = [seq[i] for i in s_idx]
            seq_names = seq_names[s_idx]
        elif self.setting.mode in ('Rolling', 'Rolling_adaptor'):
            for _ in range(self.setting.trial_number):
                trial_seq = []
                trial_seq_names = []
                for freq in self.setting.tone_freqs:
                    for amp, dB in zip(amps_in_Vrms, self.setting.GNoise_amps):
                        for loc, f_idx in zip(self.setting.stim_locs, filter_idx):
                            trial_seq.extend(((freq, amp, loc, f_idx), ) * self.setting.stim_repeat_per_trial)
                            trial_seq_names.extend(((freq, dB, loc), ) * self.setting.stim_repeat_per_trial)
                # convert to np array
                trial_seq, trial_seq_names = np.array(trial_seq), np.array(trial_seq_names, dtype=np.int32)
                # shuffling
                s_idx = np.random.permutation(trial_seq.shape[0])
                trial_seq, trial_seq_names = trial_seq[s_idx], trial_seq_names[s_idx]
                # reorganize the sequence
                seq.append((trial_seq[:, 0], trial_seq[:, 1], trial_seq[:, 2], trial_seq[:, 3]))
                seq_names.append(trial_seq_names)
        return seq, seq_names

    def _generate_noise_sequence(self):
        """
        sequence need to be a list of lists, [trials][stims in each trial]
        :return:
        """
        # TODO: use Marc's trialsmaker code to generate non-repeating sequences
        seq = []
        seq_names = []
        amps_in_Vrms = dB_to_Vrms(self.setting.GNoise_amps, self.calib_const)
        filter_idx = self._stimloc_to_filteridx(self.setting.stim_locs)
        if self.setting.mode in ('Single', 'Single_adaptor'):
            for amp, dB in zip(amps_in_Vrms, self.setting.GNoise_amps):
                for loc, f_idx in zip(self.setting.stim_locs, filter_idx):
                    seq.extend((([amp], [loc], [f_idx]), ) * self.setting.trial_number)
                    seq_names.extend((([dB], [loc]), ) * self.setting.trial_number)
            # convert names sequence into np array
            seq_names = np.array(seq_names, dtype=np.int32).squeeze()
            # shuffling
            s_idx = np.random.permutation(len(seq))
            seq = [seq[i] for i in s_idx]
            seq_names = seq_names[s_idx]
        elif self.setting.mode in ('Rolling', 'Rolling_adaptor'):
            for _ in range(self.setting.trial_number):
                trial_seq = []
                trial_seq_names = []
                for amp, dB in zip(amps_in_Vrms, self.setting.GNoise_amps):
                    for loc, f_idx in zip(self.setting.stim_locs, filter_idx):
                        trial_seq.extend(((amp, loc, f_idx), ) * self.setting.stim_repeat_per_trial)
                        trial_seq_names.extend(((dB, loc), ) * self.setting.stim_repeat_per_trial)
                # convert to np array
                trial_seq, trial_seq_names = np.array(trial_seq), np.array(trial_seq_names, dtype=np.int32)
                # shuffling
                s_idx = np.random.permutation(trial_seq.shape[0])
                trial_seq, trial_seq_names = trial_seq[s_idx], trial_seq_names[s_idx]
                # reorganize the sequence
                trial_seq = np.array(trial_seq)
                seq.append((trial_seq[:, 0], trial_seq[:, 1], trial_seq[:, 2]))
                seq_names.append(trial_seq_names)
        return seq, seq_names

    def _stimloc_to_filteridx(self, stimloc):
        """
        convert stimulus location, coded by analog channel number, to filter set index
        """
        # filter index is the same as the corresponding filter id
        if not isinstance(stimloc, Iterable):
            stimloc = [stimloc]
        achs = [spk.channel_analog for spk in self.speakers.speakers]
        idxs = []
        for s in stimloc:
            idxs.append(achs.index(s))
        return idxs

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
        # TODO
        # configure TDT device to setup stimulus parameters
        tone_param = self.tone_list[self.setting.current_trial]
        if self.setting.stim_type == 'Tone':
            self.devices['TDTs'].configure(tone_freqs=tone_param[0],
                                           tone_amps=tone_param[1],
                                           probe_chs=tone_param[2],
                                           probe_fseqs=tone_param[3],
                                           N_stims=len(tone_param[0]),
                                           filter_on=False)
        elif self.setting.stim_type == 'GNoise':
            self.devices['TDTs'].configure(WN_amps=tone_param[0],
                                           probe_chs=tone_param[1],
                                           probe_fseqs=tone_param[2],
                                           N_stims=len(tone_param[0]),
                                           filter_on=True)
        else:
            raise ValueError('stimulus type {} not known'.format(self.setting.stim_type))
        # start TDT device
        self.devices['TDTs'].start()
        # use trial timer to time and end trial
        self.trial_timer = threading.Timer(self.setting.trial_duration, self.trial_stop_fired)
        self.trial_timer.start()

    def _stop_trial(self):
        self.devices['TDTs'].pause()
        # read timing info from TDT device and save it
        RX6_timing_sig = self.devices['TDTs'].read()
        self.data.write('ImagingExp_RX6RX8_0', RX6_timing_sig)
        # put a stop frame into camera stream
        self.devices['Cam'].generate_stop_frame()
        # save and reset camera timing info
        self.data.write('{}_timing'.format(self._camera_name), np.array(self.devices['Cam']._frameTms))
        self.devices['Cam'].reset_frameInfo()
        # set trial end flags
        self.data.input_finished(('event_log', 'trial_log', 'ImagingExp_RX6RX8_0',
                                  '{}_timing'.format(self._camera_name)))
        # save data
        self.data.save(close_file=False)

    def _stop(self):
        pass

    def _pause(self):
        try:
            self.devices['TDTs'].pause()
        except AssertionError:
            pass


def cam_stream_handler(obj, stream, stop_sig=b's', data_sig=b'd', preview_sig=b'p'):
    """
    read camera stream data
    Args:
        obj: the writer.worker instance
        stream: a Stream.InputStream object
        stop_sig: byte, stop signal from the stream
        data_sig: byte, stream packets need to be saved
        preview_sig: byte, stream packets need to be skipped
    Returns:
    """
    # first check how many packets are queued up
    try:
        mr = stream.monitor()
        packet_to_read = mr[0] - stream.N_packet
        # print('latest frame: {}, current frame: {}, frames to read: {}'.
              # format(mr[0], stream.N_packet, packet_to_read))
    except TypeError:  # stream.monitor returns None because no new data arrived
        if stream.latest_packet_N is not None:
            packet_to_read = stream.latest_packet_N - stream.N_packet
            # print('No monitoring data, use latest_packet_N! latest frame: {}, current frame: {}, frames to read: {}'.
                  # format(stream.latest_packet_N, stream.N_packet, packet_to_read))
        else:
            packet_to_read = stream.Idx_initial_packet - stream.N_packet
            # print('No monitoring data, use Idx_initial_packet! latest frame: {}, current frame: {}, frames to read: {}'.
                  # format(stream.Idx_initial_packet, stream.N_packet, packet_to_read))
    if packet_to_read <= 0:
        return
    # at maximum, read 10 frames in each tick
    if packet_to_read > 10:
        packet_to_read = 10
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

    fe = ImagingExps(subject=sl.subjects[0])
    fd = fe.devices['Cam']
    ld = fe.devices['TDTs']
    # fe.start()
