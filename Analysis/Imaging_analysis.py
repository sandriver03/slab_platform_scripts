from Config import get_config
from Core.subject import load_cohort, get_cohort_names
from Analysis import image_analysis_tools as iat
from Analysis import Ultilities as au
from Ultilities import H5helpers as h5t
from GUI.VideoPlayer import play_images

import numpy as np
import tables as tl
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from collections.abc import Sized
import logging
log = logging.getLogger()


data_folder = get_config('DATA_ROOT')
RAM_LIMIT = 6     # if data is larger than 6 GB, then use temporal storage


# TODO: analysis history need to be reloaded after each finished analysis
# routines to configure analyze options
class ISI_analyze:
    """
    class to analyze intrinsic imaging (ISI) data obtained with labplatform
    """
    # class attributes
    __nosave_opts = ['save', 'recal', 'copy', 'redo_calculation', 'size_lim']
    __result_path = 'analysis_result'
    __opts_name = 'analysis_opts'
    __group_mapping = {'general': 'general_opts', 'g': 'general_opts',
                       'detrend': 'detrend_opts', 'd': 'detrend_opts',
                       'motioncorr': 'motioncorr_opts', 'm': 'motioncorr_opts',
                       'tempfilt': 'tempfilt_opts', 't': 'tempfilt_opts',
                       'spatfilt': 'spatfilt_opts', 's': 'spatfilt_opts',
                       'dfoverf': 'dFoverF_opts', 'dFoverF': 'dFoverF_opts', 'f': 'dFoverF_opts',
                       'plot': 'plot_opts', 'p': 'plot_opts',
                       'save': 'save_opts', 'sv': 'save_opts'
                       }
    __temp = 'temp'
    __groups = ('general', 'motioncorr', 'detrend', 'tempfilt', 'spatfilt', 'dFoverF', 'save', 'plot')

    def __init__(self, file=None):
        """
        Args:
            file: tables.file.File instance, handle to the data to be analyzed or str, path to the file
        """
        self.file = file
        self.data_node = None
        self.file_handle = None

        # to be safe, create a separate result file to store result from individual trials
        self.result_indtrial_file = None
        self.result_indtrial_fh = None
        self.result_group = None    # __result_path
        self.result_indtrial_node = None   # directly on the root node

        # get a pointer to the analysis result ready to plot
        self.result_node = None

        # if the analysis routine is using the temporal file storage
        self._using_temp = False

        self.general_opts = {'node': '',
                             'redo_calculation': False,
                             'down_sample': 2,
                             'cond_alt': 'tone_sequence',
                             'cond': 'stim_name_sequence',
                             'LED_chs': 'ImagingExp_RX6RX8_0_Setting_channels_touse',
                             'reference_ch': 1,
                             'data_node': 'PrimeCam_0_PrimeCam_0_tcp',
                             'timing_node': 'PrimeCam_0_timing',
                             'trigger_node': 'ImagingExp_RX6RX8_0',
                             'copy': False,
                             'chs_toanalyze': None,
                             'Rolling_name': 'rolling_trials',     # condition name for rolling experiments
                             'default_trial_duration': 10,         # when trial_duration is not in the data file
                             }
        self.detrend_opts = {'enable': False,
                             'model': 'polynomial',
                             'order': 1,
                             'keep_mean': True,
                             'size_lim': 2048,
                             'copy': False,
                             'save': False,
                             }
        self.motioncorr_opts = {'enable': False,
                                'manual': False,
                                'upsample_factor': 100,
                                'model': 'rigid',
                                'copy': False,
                                'recal': False,
                                'save': False
                                }
        self.tempfilt_opts = {'enable': False,
                              'kernel': 'median',
                              'size': 5,
                              'copy': False,
                              'save': False
                              }
        self.spatfilt_opts = {'enable': True,
                              'sigma': 1.3,
                              'kernel': 'gaussian',
                              'copy': False,
                              'save': False
                              }
        self.dFoverF_opts = {'tm_wd': 0.,
                             'baseline_wd': 3.,
                             'dt': 1.0
                             }
        self.plot_opts = {'plot': True,
                          'chs_toplot': None,
                          'chs_clim': {
                              0: (-8e-4, 8e-4),    # red channel
                              1: (-2e-3, 2e-3),    # green channel
                              2: (-5e-2, 5e-2),    # blue channel
                                       },
                          'conds_toplot': None
                          }
        self.save_opts = {'save_indtrial': True,   # save dF/F for each trial
                          'save_plot': True        # save the trial average plot
                          }

        # temporal info path
        self._temp_path = self.__temp

        # analysis history
        self.analysis_history = []

        # parameters related to analysis
        self.N_trials = 0
        self.conditions = None
        self.cond_indtrial = None
        self.trial_duration = 0    # only needed to determine average result size
        self.inter_trial_interval = 0
        self.exp_mode = None

        self._current_trial = None  # current trial index to be analyzed
        self._current_cond = None  # current condition being analyzed
        self._current_cond_idx = None  # indexes for trials in current condition

        self.channels = [0, ]
        self.reference_image = None  # reference image used for motion correction as well as display
        self.mask = None             # mask used in case of masked motion correction

        self._mismatch = []           # keep track of timing vs. images mismatch
        self._bad_frames = []         # frames contain incorrect channel info
        self._current_imgdata = None  # current image data to be analyzed
        self._current_tmdata = None   # current timing data
        self._current_trgdata = None  # current trigger info
        self._current_img_indch = None  # for debug use
        self._current_tm_indch = None   # for debug use
        self._dff_avgs = None           # temporal storage for average dff across trials data
        self._dff = None                # temporal storage for single trial dff data
        self._favg = None               # temporal storage for average frame intensities in one trial
        self._resROI = None             # ROI for the response

        # load file if provided
        if self.file:
            self.load_file(self.file)

    def configure(self, opts_group, **kwargs):
        """
        configure analyze options
        Args:
            opts_group: str, one of the self.groups
            **kwargs: arguments to be updated
        Returns:
            None
        """
        opts_toconfig = getattr(self, self.__group_mapping[opts_group.lower()])
        # update options with values from kwargs
        for k in kwargs.keys():
            if k in opts_toconfig.keys():
                opts_toconfig.update({k: kwargs[k]})
            else:
                log.warning('key {} is not in the analysis options for group {}. No change is made.'.
                            format(k, self.__group_mapping[opts_group]))

    def load_file(self, file=None):
        """
        access a .h5d file which contains data to one ISI experiment
        Args:
            file: str or tables.File handle, the h5d file to be accessed
        Returns:
            None
        """
        if file is None:
            file = self.file
        self._access_file(file)
        # load analysis history
        self._load_history()
        # check if the temp and result group are there. if not create them
        _ = h5t.get_or_append_node(self.data_node, self._temp_path)
        _ = h5t.get_or_append_node(self.data_node, self.__result_path)

        # load basic experiment information
        self._get_experiment_info()

    def set_reference_image(self, image):
        """
        set a reference image to be used in the analysis
        Args:
            image: 2D np array
        Returns:
            None
        """
        self.reference_image = image

    def analyze(self, file=None):
        """
        start ISI analysis
        Args:
            file: str or tables.File handle, the h5d file to be accessed
        Returns:
            None
        """
        # load file if it is provided; otherwise assume it is already loaded
        if file:
            self.load_file(file)
        # check if the same analysis is already done
        opts_ = self._get_tosave_opts()
        _run_analysis = True
        if opts_ in self.analysis_history:
            _run_analysis = False
            # when using motion correction, need to make sure the mask image is not custom chosen
            if opts_['motioncorr']['enable'] and opts_['motioncorr']['manual']:
                _run_analysis = True

        if _run_analysis:
            # start a new analysis
            # prepare node to store the data
            self.result_node = self._generate_result_node()
            # to be safe, save the analysis options as the group attribute
            self.result_node._f_setattr('opts', opts_)
            analysis_idx = len(self.analysis_history)

            # set file path for saving individual trials data into a separate file
            if self.save_opts['save_indtrial']:
                self.result_indtrial_file = self._generate_indtrial_filename()

            # experiment info is already loaded when accessing file
            # set the reference image if not set
            if self.reference_image is None:
                self._get_reference_image()

            # check which steps need to be done, which intermediate results can be loaded
            # currently, only the motion correction has intermediates that can be loaded
            motioncorr_idx = None
            if self.motioncorr_opts['enable']:
                hist_motion = [opts['motioncorr'] for opts in self.analysis_history]
                motioncorr_idx = hist_motion.index(self.motioncorr_opts)

            # loop over different conditions
            if self.channels:
                # 'Single' and 'Rolling' mode should be dealt differently
                if self.exp_mode == 'Single':
                    for cond in self.conditions:
                        self._current_cond = cond
                        inter_val = np.equal(self.cond_indtrial, cond)
                        cond_dim = np.array(self.cond_indtrial.shape[1:])
                        # condition array could be any shape, but first dimension is always trials
                        cond_idx = np.where(inter_val.sum(tuple(range(1, len(inter_val.shape)))) ==
                                            cond_dim.prod())[0]
                        self._current_cond_idx = cond_idx
                        # analyze a single condition
                        try:
                            cal_avg = self.general_opts['calculate_avg']
                        except KeyError:
                            cal_avg = True
                        if cal_avg:
                            # create array to hold the result
                            self._dff_avgs = self._prepare_result_array(self.trial_duration,
                                                                        self.dFoverF_opts['tm_wd'],
                                                                        self.dFoverF_opts['dt'],
                                                                        self._get_image_shape(),
                                                                        self.general_opts['down_sample'],
                                                                        len(self.channels))
                        _ = self.analyze_single_cond(cond, cond_idx, calculate_avg=cal_avg,
                                                     save=True, chs=self.channels, avg_array=self._dff_avgs)
                        # plot average result
                        if self.plot_opts['plot']:
                            self.plot_single_cond(cond, self.plot_opts['chs_toplot'], analysis_idx,
                                                  data=self._dff_avgs,
                                                  clim=self.plot_opts['chs_clim'],
                                                  save=self.save_opts['save_plot'])
                elif self.exp_mode == 'Rolling':
                    cond_idx = list(range(self.cond_indtrial.shape[0]))
                    self._current_cond_idx = cond_idx
                    # analyze every trial; do not calculate trial averages
                    cond_name = self.general_opts['Rolling_name']
                    self.analyze_single_cond(cond_name, cond_idx, calculate_avg=False,
                                             save=True, chs=self.channels)
                    # no plotting as well
                else:
                    raise ValueError('experiment mode: {} not known'.format(self.exp_mode))

        else:
            # load existing analysis result
            index = self.analysis_history.index(opts_)
            self._get_result_node(index)
            # plot results
            if self.channels and self.plot_opts['plot'] and self.exp_mode == 'Single':
                if self.plot_opts['conds_toplot'] is None:
                    conds_toplot = self.conditions
                else:
                    conds_toplot = self.plot_opts['conds_toplot']
                if self.plot_opts['chs_toplot'] is None:
                    chs_toplot = self.channels
                else:
                    chs_toplot = self.plot_opts['chs_toplot']
                for cond in conds_toplot:
                    for ch in chs_toplot:
                        try:
                            clim = self.plot_opts['chs_clim'][ch]
                        except KeyError:
                            clim = (-8e-4, 8e-4)
                        self.plot_result(cond=cond, channel=ch, analysis_index=index,
                                         save=self.save_opts['save_plot'], clim=clim)

    def plot_result(self, cond=None, channel=0, analysis_index=None, data_prefix='dff_avg',
                    data=None, cmap='gray', clim=(-8e-4, 8e-4),
                    save=None, save_dir=None):
        """
        plot analysis result in an interactive GUI
        Args:
            cond: condition to be plotted, see self.conditions
            channel: int, which channel to be plotted
            analysis_index: int, which analysis to plot
            data_prefix: str, prefix to earray name. for ISI result, it is 'dff_avg'
            data: 3-D image time series, if provided will be plotted instead of loading from cond and channel
            cmap: str or a colormap can be used by plt.imshow
            clim: color limits, see plt.imshow
            save: bool, if save the plotted figures
            save_dir: path or str, where to save the plotted result
        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        if cond is None:
            try:
                cond = self.conditions[0]
            except TypeError:
                raise ValueError('the condition to be plotted cannot be decided')

        # get data
        if data is None:
            data, fig_name = self.prepare_result_to_plot(cond, channel, analysis_index,
                                                         load_data=True, data_prefix=data_prefix)
        else:
            _, fig_name = self.prepare_result_to_plot(cond, channel, analysis_index,
                                                      load_data=False, data_prefix=data_prefix)

        if save is None:
            save = self.save_opts['save_plot']
        if save:
            if save_dir is None:
                # save in the same folder as the data
                save_dir = os.path.split(self.file)[0]
        else:
            save_dir = None

        # plot with GUI.VideoPlayer.play_images
        play_images(data, title=fig_name, cmap=cmap, clim=clim, save_dir=save_dir)

    def prepare_result_to_plot(self, cond, channel, analysis_index, data_prefix='dff_avg', load_data=True):
        """
        load data and generate figure name for plotting
        :param cond: array like, experiment condition to be loaded
        :param channel: int, channel to be loaded
        :param analysis_index: int, analysis index to be loaded
        :param data_prefix: str, earray naming prefix
        :param load_data: bool, if load the data from associated .h5 file
        :return:
        """
        # result earray name
        earray_name = data_prefix + '_condition_{}_channel_{}'.format(self._get_cond_str(cond), channel)

        if load_data:
            if self.result_node is None:
                self.result_node = self._get_result_node(analysis_index)
            result_node = self.result_node
            data = result_node[earray_name]
        else:
            data = None

        exp_name = os.path.split(self.file)[1].split('.h5')[0]
        fig_name = exp_name + '_' + earray_name + '_analysis' + str(analysis_index)
        return data, fig_name

    def analyze_single_cond(self, cond, trial_idxs, calculate_avg, save,
                            chs=None, avg_array=None, keep_indtrial=False):
        """
        analyze all trials provided in trial_idxs
        :param cond: list, condition to be analyzed
        :param trial_idxs: iterable, indexes of trials to be analyzed
        :param chs: channels to be analyzed
        :param calculate_avg: bool, if also calculate averages from all the trials in the condition
        :param save: bool, if save result into .h5 file
        :param avg_array: np array to hold the average result; must be provided if calculate_avg is True
        :param keep_indtrial: bool, if also return results from each individual trial. may consume a lot of RAM
        :return:
        """
        if chs is None:
            chs = self.channels
        if not isinstance(chs, Sized):
            chs = [chs]
        dff_indtrial = []
        favg_indtrial = []

        # prepare arrays to hold average dff across trials
        if calculate_avg:
            if avg_array is None:
                raise ValueError('to calculate average response, the array to hold the result must be provided')
            self._dff_avgs = avg_array

        # analyze individual trials
        for trial in trial_idxs:
            _dff, _favg = self._analyze_indtrial(trial)
            if keep_indtrial:
                dff_indtrial.append(_dff)
                favg_indtrial.append(_favg)
            self._dff, self._favg = _dff, _favg

            # save ind trial results; also add ind trial dff to dff_avgs
            if save:
                self._save_trial_result(trial, cond, chs, self._dff, self._favg)

            if calculate_avg:
                # update average dff array for each condition
                self._dff_avgs = self._update_cond_average(self._dff, self._dff_avgs, chs,
                                                           len(trial_idxs), self._current_trial)

        if save:
            # average dff array only need to be saved at the end
            if calculate_avg:
                self._save_average_result(cond, chs, len(trial_idxs), self._dff_avgs)
            # save analysis options as analysis history
            self._write_history()
            self.file_handle.flush()

        return self._dff_avgs, dff_indtrial, favg_indtrial

    def plot_single_cond(self, condition, channel, analysis_idx, clim, save, data=None):
        """
        plot the average result as a time series of frames for a single condition
        :param condition: array like
        :param channel: int, channel to be plotted
        :param analysis_idx: int
        :param clim: colormap limits, 2-element
        :param save: bool, if save resulting figures
        :param data: data to be plotted; if provided then no data loading is needed
        :return:
        """
        if channel is None:
            channel = self.channels
        if not isinstance(channel, Sized):
            channel = [channel]

        for ch in channel:
            ch_idx = channel.index(ch)
            try:
                ch_clim = clim[ch]
            except (KeyError, TypeError):   # type error in case clim is None
                ch_clim = (-8e-4, 8e-4)
            self.plot_result(cond=condition, channel=ch, analysis_index=analysis_idx,
                             save=save, clim=ch_clim, data=data[ch_idx])
        self.plot_reference_image()

    @staticmethod
    def _get_cond_str(cond):
        if not isinstance(cond, Sized) or isinstance(cond, str):
            return str(cond)
        res = str(cond[0])
        for ele in cond[1:]:
            res += '_' + str(ele)
        return res

    # Todo: problematic, need to fix it
    def _get_reference_image(self):
        """
        load reference image for the experiment
        Returns:
            None
        """
        try:
            self.reference_image = self.data_node.snapshot[-1]
        except (tl.exceptions.NoSuchNodeError, IndexError):
            trial_node = getattr(self.data_node, self._get_trial_node_name(0))
            data_name = self.general_opts['data_node']
            try:
                # take the first image from the reference channel
                refch_idx = self.channels.index(self.general_opts['reference_ch'])
                self.reference_image = trial_node[data_name][refch_idx]
            except ValueError:
                # no reference channel images, return None
                return None

    def plot_reference_image(self, data=None, cmap='gray', clim=None, save=True,
                             save_dir=None, file_name=None):
        """
        plot and/or save reference image
        Args:
            data: np array, image to be plotted
            cmap: str or a colormap can be used by plt.imshow
            clim: color limits, see plt.imshow
            save: bool, if save the plotted figures
            save_dir: path or str, where to save the plotted result
            file_name: str, name of saved file
        Returns:
            None
        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if save is None:
            save = self.save_opts['save_plot']
        if save:
            if save_dir is None:
                # save in the same folder as the data
                save_dir = os.path.split(self.file)[0]
        else:
            save_dir = None

        if data is None:
            # if no reference image, return immediately without doing anything
            if self.reference_image is None or not np.any(self.reference_image):
                return
            data = self.reference_image
        if len(data.shape) == 2:
            data = data.reshape((1, ) + data.shape)

        if file_name is None:
            exp_name = os.path.split(self.file)[1].split('.h5')[0]
            file_name = exp_name + '_reference'
        play_images(data, title=file_name, cmap=cmap, clim=clim, save_dir=save_dir)

    def _get_result_node(self, analysis_index=None):
        """
        get the .h5 node which contains the analysis results for the analysis specified by analysis_index
        Args:
            analysis_index: int
        Returns:
            None
        """
        if analysis_index is None:
            analysis_index = len(self.analysis_history) - 1
        self.result_node = self.result_group[self._generate_result_node_name(analysis_index)]

    def _get_result_indtrial_node(self, analysis_index=None, mode='a'):
        """
        get the .h5 node which contains the analysis results for the analysis specified by analysis_index
        Args:
            analysis_index: int
            mode: str, mode to open the file
        Returns:
            None
        """
        folder, data_file = os.path.split(self.file)
        if analysis_index is None:
            result_appendix = self._generate_result_node_name(len(self.analysis_history) - 1)
        else:
            result_appendix = self._generate_result_node_name(analysis_index)
        result_file = data_file.split('.')[0] + '_' + result_appendix + '.h5'
        result_file = os.path.join(folder, result_file)
        self.result_indtrial_file = result_file
        # open the file, and get the node points to the result node
        self.result_indtrial_fh = h5t.get_file_handle(result_file, mode=mode)
        self.result_indtrial_node = self.result_indtrial_fh.root

    def select_response_ROI(self, data, resp_wd=(4, 6), meth='manual',
                            cmap='gray', clim=(-8e-4, 8e-4)):
        """
        select a ROI for the response, manually or automatically
        Args:
            data: 3-D image of analysis result, 1st dimension is time
            resp_wd: 2 elements list or tuple, time window of the response in second
            meth: str 'manual' or 'auto', method used to select the ROI
            cmap: matplotlib colormap, used to dislay image
            clim: range of colormap mapping, used to display image
        Returns:
            2-D numpy array, mask for the ROI
        """
        if meth == 'manual':
            # check if the data is already the correct image, or a 3-D time series of images
            if data.shape.__len__() == 3:
                dF_opts = self.result_node._v_attrs.opts['dFoverF']
                # get frame index for the response window
                frame_dt = dF_opts['dt']
                if dF_opts['tm_wd']:
                    resp_idx = self._wd_to_idx(np.array(resp_wd) - dF_opts['tm_wd'][0], frame_dt)
                else:
                    resp_idx = self._wd_to_idx(resp_wd, frame_dt)
                resp_image = np.mean(data[resp_idx, :, :], 0)
            elif data.shape.__len__() == 2:
                resp_image = data
            # manually select ROI
            return iat.select_roi(resp_image, manual='True', cmap=cmap, clim=clim)
        elif meth == 'auto':
            # TODO
            # calculate dprime
            raise NotImplementedError

    @staticmethod
    def _wd_to_idx(wd, dt):
        """
        convert time windown range to indexes
        Args:
            wd: window range
            dt: time step
        Returns:
            list of indexes
        """
        if not isinstance(wd, Sized):
            bs_wd = [wd]
        else:
            bs_wd = wd
        if len(bs_wd) < 2:
            bs_wd = list(bs_wd)
            bs_wd.insert(0, 0)
        return np.arange(bs_wd[0]/dt, bs_wd[1]/dt, dtype=np.int32)

    def _calculate_dprime(self, data, opts=None, resp_wd=None, save=False,
                          save_name=None, analysis_index=None):
        """
        calculate 2D d prime map in regards to baseline using individual trial averages
        Args:
            data: 4-D image of analysis result across different trials, 1st dimension is trial, 2nd dimension time
            opts: dict, must contains keys baseline_wd, dt
            resp_wd: 2 elements sized; if provided, only return average dprime in this time window
            save: bool, if save calculated result
            save_name: str
            analysis_index: int or None, which analysis the dprime data belongs to
        Returns:
            3D np.ndarray, dprime for each pixel across time
        """
        if isinstance(data, tl.earray.EArray):
            opts_analysis = data._v_parent._v_attrs['opts']
            if opts is None:
                opts = opts_analysis['dFoverF']
            if save_name is None:
                save_name = data.name[data.name.index('condition'):]
            analysis_index = self.analysis_history.index(opts_analysis)
        else:
            if analysis_index is None:
                log.warning('using currently loaded analysis options...')
                opts_analysis = self.result_indtrial_node._v_attrs['opts']
                analysis_index = self.analysis_history.index(opts_analysis)
            else:
                opts_analysis = self.analysis_history[analysis_index]
            if opts is None:
                opts = opts_analysis['dFoverF']
            if save_name is None:
                save_name = 'undefined'
                if save:
                    log.warning('node name to be saved is not specified')

        # calculate mean and variance from baseline
        bs_idx = self._wd_to_idx(opts['baseline_wd'], opts['dt'])
        bs_data = data[:, bs_idx, :, :]
        bs_mean = np.mean(bs_data, (0, 1))
        bs_var = np.var(bs_data, (0, 1))
        # construct result
        result = np.zeros(data.shape[1:])
        for t in range(data.shape[1]):
            t_data = data[:, t, :, :]
            t_mean = np.mean(t_data, (0, 1))
            t_var = np.var(t_data, (0, 1))
            result[t] = np.abs(t_mean - bs_mean) / (np.sqrt((t_var + bs_var) / 2))
        # if save, save the result here
        if save:
            self._save_dprime_result(result, save_name, analysis_index)
        # if response window is provided, only return averaged result in response window
        if resp_wd is not None:
            resp_idx = self._wd_to_idx(resp_wd, opts['dt'])
            result = np.mean(result[resp_idx], 0)
        return result

    def _save_dprime_result(self, dprime_data, name, analysis_index=None):
        """
        save calculated dprime data into .h5 file
        Args:
            dprime_data: np.ndarray, dprime result
            name: str, name of the node to be saved
            analysis_index: int, point to the analysis node to be saved
        Returns:
            None
        """
        if analysis_index is None:
            analysis_index = len(self.analysis_history) - 1
        node_name = self._generate_result_node_name(analysis_index)
        result_name = 'dprime_' + name
        ext_file_name = self._generate_indtrial_filename(analysis_index)
        # get the immediate path to save the data in the main file
        node_tosave = self.data_node[self.__result_path][node_name]
        _ = node_tosave._v_file.create_earray(node_tosave, result_name, obj=dprime_data)
        self.file_handle.flush()
        # save also in the indtrial file
        if not self.result_indtrial_node or \
            self.result_indtrial_file != self._generate_result_node_name(analysis_index):
            self._get_result_indtrial_node(analysis_index)
        _ = self.result_indtrial_fh.create_earray(self.result_indtrial_fh.root,
                                                  result_name, obj=dprime_data)
        self.result_indtrial_fh.flush()

    def get_response_timecourse(self, data, mask):
        """
        get indivual trial time course of the signal specified by mask. use loaded individual trial result
        file
        Args:
            data: tables.Earray, str, np.ndarray, image data to be selected; expected to be 4D
            mask: binary image of same size as data, or 2-elements tuple or list
        Returns:
            np.ndarray
        """
        if not self.result_indtrial_fh or not self.result_indtrial_fh.isopen:
            raise RuntimeError('file for individual trial result is not specified. use ._get_result_indtrial_node')
        # input checking
        if isinstance(data, (tl.earray.EArray, np.ndarray, tl.vlarray.VLArray)):
            pass
        elif isinstance(data, str):
            data = getattr(self.result_indtrial_node, data)
        else:
            raise ValueError('data with class: {} cannot be processed'.format(data.__class__))
        if isinstance(mask, (tuple, list)):
            if len(mask) != 2:
                raise ValueError('expected indexes in form of (rs, cs)')
            pixel_idxs = mask
        elif isinstance(mask, np.ndarray):
            if mask.shape != data.shape[-2:]:
                raise ValueError('mask shape: {} does not match data shape: {}'.format(
                    mask.shape, data.shape[-2:]))
            else:
                pixel_idxs = np.where(mask)

        # get data
        if isinstance(data, tl.earray.EArray):
            # if file is small enough, read it first into ram
            data_in_ram = None
            if np.prod(data.shape) * data.dtype.itemsize / 1024 ** 3 < RAM_LIMIT:
                data_in_ram = data[:]
            if data_in_ram is not None:
                roidata = data_in_ram[:, :, pixel_idxs[0], pixel_idxs[1]].copy()
            else:
                roidata = np.dstack([data_in_ram[:, :, i, j] for i, j in zip(*pixel_idxs)])
        elif isinstance(data, np.ndarray):
            roidata = data[:, :, pixel_idxs[0], pixel_idxs[1]].copy()
        return roidata

    def _analyze_indtrial(self, trial_idx):
        """
        perform analysis on individual trials
        Args:
            trial_idx: int, index of the trial to be analyzed
        Returns:
            analysis result
        """
        self._load_trial_data(trial_idx)
        # prepare the data to be analyzed
        temp_group = h5t.get_or_append_node(self.data_node, self._temp_path)
        img_data, is_ndarray = au._prepare_data(self._current_imgdata, temp_path=temp_group._v_pathname,
                                                copy=self.general_opts['copy'],
                                                down_scale=self.general_opts['down_sample'])
        self._using_temp = not is_ndarray
        # split the data into individual channels
        img_indchs, tm_indchs, bad_frs = iat.split_data_on_chs(img_data, self._current_tmdata,
                                                               len(self.all_channels), self.all_channels)
        # only get those channels to be analyzed
        img_to_analyze, tm_to_analyze = [], []
        for ch in self.channels:
            ch_idx = self.all_channels.index(ch)
            img_to_analyze.append(img_indchs[ch_idx])
            tm_to_analyze.append(tm_indchs[ch_idx])
        self._bad_frames.append(bad_frs)
        self._current_img_indch = img_to_analyze
        self._current_tm_indch = tm_to_analyze

        # start analysis
        if self.motioncorr_opts['enable']:
            pass

        # detrend
        if self.detrend_opts['enable']:
            # TODO: seems detrend will help a lot
            pass

        # temporal filtering
        if self.tempfilt_opts['enable']:
            pass

        # spatial filtering
        if self.spatfilt_opts['enable']:
            for img_idx in range(img_to_analyze.__len__()):
                img_to_analyze[img_idx] = iat.image_spatial_filter(img_to_analyze[img_idx], **self.spatfilt_opts)

        # calcualte dF/F, on different channels
        dff_allchs = []
        F_avg_allchs = []
        for img, tm in zip(img_to_analyze, tm_to_analyze):
            dff, F_avg = iat.deltaF_over_F(img, tm, 0, **self.dFoverF_opts)
            dff_allchs.append(dff[0])
            F_avg_allchs.append(F_avg[0])
            # clean-up temporal storage if used
            if self._using_temp:
                # TODO: not analyzed channels need to be removed
                # in this case img_data should be a node in pytables
                self.file_handle.remove_node(img)
        # print('trial: {} \n'.format(trial))
        # print('result shape: {} \n'.format(dff[0].shape))
        # print('timing info: {}, {} \n'.format(tm_data[0], tm_data[-1]))

        return dff_allchs, F_avg_allchs

    def _load_trial_data(self, trial_idx=None):
        """
        load trial data
        Args:
            trial_idx: int, index of the trial
        Returns:
            None
        """
        if trial_idx is not None:
            self._current_trial = trial_idx
        if self._current_trial is None:
            raise ValueError('the trial to be loaded is not specified')

        log.info("loading data from trial {}...".format(trial_idx))
        trial_name = self._get_trial_node_name(self._current_trial)

        # get data as a pytables.earray
        trial_node = self.data_node[trial_name]
        img_data = trial_node[self.general_opts['data_node']]
        # timing and trigger data are read into RAM
        tm_data = trial_node[self.general_opts['timing_node']][:]
        trig_data = trial_node[self.general_opts['trigger_node']][:]

        # timing data's unit need to be changed to second
        tm_data = au.primecam_convert_timing_unit(tm_data)
        # TODO: need to check if the timing data matches the image data. not sure what happens if not
        if len(tm_data) > len(img_data):
            tm_data = tm_data[:len(img_data)]
            self._mismatch.append((trial_idx, len(img_data), len(tm_data)))

        self._current_imgdata = img_data
        self._current_tmdata = tm_data
        self._current_trgdata = trig_data

    def _update_cond_average(self, single_trial_res, avg_array, channels, N_trial, trial=None):
        """
        update average response for each condition array, with a single trial result
        :param single_trial_res: list of np array, response from a single trial
        :param avg_array: list of np array, average response for the given condition
        :param channels: array like, light channels
        :param N_trial: int, number of total trials
        :param trial: int, current trial
        :return:
        """
        if trial is None:
            trial = self._current_trial
        for ch in channels:
            ch_idx = channels.index(ch)
            # dF/F
            # put data into averages, check the shape first
            if single_trial_res[ch_idx].shape != avg_array[ch_idx].shape:
                log.warning('trial {} returned result with shape {}'.
                            format(trial, single_trial_res[ch_idx].shape))
                single_trial_res[ch_idx] = self._adjust_result_size(single_trial_res[ch_idx],
                                                                    avg_array[ch_idx].shape)
            avg_array[ch_idx] = avg_array[ch_idx] + single_trial_res[ch_idx] / N_trial
        return avg_array

    def _save_trial_result(self, trial, cond, channels, dff, frame_avg):
        """
        save results from single trial analysis
        :param trial: index of current trial to be saved
        :param cond: experiment condition, array like
        :param channels: channels in the data
        :param dff: dF/F result, 3D np array
        :param frame_avg: average intensity of each frame, should be 2D
        :return:
        """
        F_avg = frame_avg
        # save results
        # only analyzed channels are in dff_avgs and F_avg
        for ch in channels:
            ch_idx = channels.index(ch)

            # average intensity for each frame in each condition
            # save in main file
            favg_earray_name = 'favg_condition_{}'.format(self._get_cond_str(cond), str(ch))
            favg_node = h5t.get_or_append_node(self.result_node, favg_earray_name, 'earray',
                                               atom=tl.Atom.from_dtype(np.dtype(float)),
                                               shape=(0, 5))
            f_data = trial * np.ones((len(F_avg[ch_idx]), 5))
            f_data[:, 1] = channels[ch_idx]
            f_data[:, 2:] = F_avg[ch_idx]
            favg_node.append(f_data)

            # save individual trial results into a separate file if needed
            if self.save_opts['save_indtrial']:
                # prepare saving path
                if not self.result_indtrial_fh or not self.result_indtrial_fh.isopen:
                    self.result_indtrial_fh = tl.open_file(self.result_indtrial_file, mode='w')
                self.result_indtrial_node = self.result_indtrial_fh.root
                self.result_indtrial_node._f_setattr('opts', self._get_tosave_opts())
                # dff individual trial
                # TODO: change to vlarray
                res_earray_name = 'dff_indtrial_condition_{}_channel_{}'.format(self._get_cond_str(cond), str(ch))
                res_node = h5t.get_or_append_node(
                    self.result_indtrial_node, res_earray_name, 'vlarray',
                    atom=tl.Atom.from_type(dff[ch_idx].dtype.name, shape=dff[ch_idx].shape[1:])
                )
                res_node.append(dff[ch_idx])
                # f_avg
                favg_node = h5t.get_or_append_node(self.result_indtrial_node, favg_earray_name, 'earray',
                                                   atom=tl.Atom.from_dtype(np.dtype(float)),
                                                   shape=(0, 5))
                favg_node.append(f_data)

            # to be safe, flush after each trial
            self.result_indtrial_fh.flush()
            self.file_handle.flush()

    def _save_average_result(self, cond, channels, n_trials, res_avg):
        """
        save all trials average result for each condition
        :param cond: condition, array like
        :param channels: channels in dataset, array like
        :param n_trials: number of trials in current condition
        :param res_avg: result, 3D np array
        :return:
        """
        """
        save all trials average result for each condition
        Returns:
            None
        """
        dff_avgs = res_avg
        # calculate average dff for each condition
        for ch in channels:
            ch_idx = channels.index(ch)
            res_name = 'dff_avg_condition_{}_channel_{}'.format(self._get_cond_str(cond), str(ch))
            avg_data = dff_avgs[ch_idx]
            # save all trials average, to both main file and separate file
            _ = self.file_handle.create_earray(self.result_node, res_name, obj=avg_data)
            if self.save_opts['save_indtrial']:
                _ = self.result_indtrial_fh.create_earray(self.result_indtrial_node, res_name, obj=avg_data)

        # flush file
        self.file_handle.flush()
        if channels and self.save_opts['save_indtrial']:
            self.result_indtrial_fh.flush()

    def _prepare_result_array(self, trial_dur, anal_wd, delta_t, data_shape, ds_ratio, n_chs):
        """
        prepare the (list of) np array to hold the analysis result
        :param trial_dur: trial duration, s
        :param anal_wd: analysis window for each trial, [start, end]
        :param delta_t: time step for each frame, s
        :param data_shape: shape of each image, [n_row, n_column]
        :param ds_ratio: down sample ration, int. Note: resulting shape should also be integer.
            this is NOT checked!
        :param n_chs: int, number of channels
        :return:
            list of np array to hold the results
        """
        if trial_dur is None:
            trial_dur = self.general_opts['default_trial_duration']
        # get correct array size from tiral_duration and dt, as well as image size and scale factor
        # Nbins need to be decided by either trial_duration, or if specified, dFoverF_opts['tm_wd']
        analysis_wd = self._get_timewindow(trial_dur, anal_wd)
        Nbins = int(np.ceil(np.diff(analysis_wd) / delta_t))
        res_shape = np.array((Nbins,) + data_shape[1:], dtype=np.int32)
        if ds_ratio > 1:
            res_shape[1:] = res_shape[1:] / ds_ratio
        # need to create one array for each channel
        dff_avgs = []
        for _ in range(n_chs):
            dff_avgs.append(np.zeros(res_shape, dtype=float))
        return dff_avgs

    def _generate_indtrial_filename(self, analysis_index=None):
        """
        get the correct file name to be used to store individual trial analysis result
        Returns:
            str
        """
        # self.file is an absolute path
        folder, data_file = os.path.split(self.file)
        if analysis_index is None:
            result_appendix = self._generate_result_node_name(len(self.analysis_history))
        else:
            result_appendix = self._generate_result_node_name(analysis_index)
        result_file = data_file.split('.')[0] + '_' + result_appendix + '.h5'
        result_file = os.path.join(folder, result_file)
        # validate if the file already exists
        if os.path.isfile(result_file):
            log.warning("There is already a file named '{}'".format(result_file))
        return result_file

    def _generate_result_node(self):
        """
        generate the node path to save current analysis result
        Returns:
            tables.group
        """
        return h5t.get_or_append_node(self.data_node[self.__result_path],
                                      self._generate_result_node_name(len(self.analysis_history)))

    @staticmethod
    def _adjust_result_size(dff_indtrial, result_shape):
        """
        make sure result size is the same as the expected size
        Returns:
            np array
        """
        if dff_indtrial.shape[0] > result_shape[0]:
            return dff_indtrial[:result_shape[0]]
        else:
            data = np.zeros(result_shape)
            data[:dff_indtrial.shape[0]] = dff_indtrial
            return data

    def _get_image_shape(self, cam_name='PrimeCam_0'):
        """
        for imaging experiment, get the shape of each image
        Returns:
            tuple
        """
        attr_name = cam_name + '_Setting_shape'
        image_shape = self.data_node._v_attrs[attr_name]
        # if software scaling is specified
        sf_str = cam_name + '_Setting_software_scaling'
        if sf_str in self.data_node._v_attrs:
            ss = self.data_node._v_attrs[sf_str]
            if not isinstance(ss, (list, tuple)):
                # scaling is a number
                ss = list([ss])
            if isinstance(ss, tuple):
                ss = list(ss)
            # scaling the shape parameter, starting from 2nd position; 1st should always be -1
            old_shape = list(image_shape[1:])
            if len(ss) == 1:
                new_shape = [int(old_val/ss[0]) for old_val in old_shape]
            elif len(ss) == len(old_shape):
                new_shape = [int(val[0]/val[1]) for val in zip(old_shape, ss)]
            image_shape = (-1, ) + tuple(new_shape)
        return image_shape

    def _get_device_names(self):
        """
        get the name of devices used in the experiment
        Returns:
            tuple of str
        """
        names = [k for k in self.data_node._v_attrs.__dict__.keys() if not k.startswith('_') and 'Setting' in k]
        res = []
        for n in names:
            if n.split('_Setting')[0] not in res:
                res.append(n.split('Setting')[0])
        return tuple(res)

    @staticmethod
    def _get_trial_node_name(trial_idx):
        """
        given a trial index (or trial number), return the name of the trial as used in the h5d file
        Args:
            trial_idx: int
        Returns:
            str
        """
        return 'trial_' + str(trial_idx).zfill(get_config('FILL_DIGIT'))

    @staticmethod
    def _generate_result_node_name(analysis_idx):
        """
        given an analysis index (or analysis number), get the group name to hold current analysis result
        Args:
            trial_idx: int
        Returns:
            str
        """
        return 'result_' + str(analysis_idx).zfill(get_config('FILL_DIGIT'))

    def _access_file(self, file):
        """
        access file with experiment data
        Args:
            file: str or tables.File, tables.Group
        Returns:
            None
        """
        self._close_file()
        # access the data in the .h5 file
        # make sure the file is absolute path if it is str; assume in this case the str is relative to data_folder
        if isinstance(file, str) and not os.path.isabs(file):
            file = os.path.join(data_folder, file)
        if isinstance(file, str):
            # first check if the file is already opened
            fh = h5t.get_handler_by_name(file)
            if not fh and os.path.isfile(file):
                fh = tl.open_file(file, mode='a')
        elif isinstance(file, (tl.File, tl.Group)):
            fh = file
        else:
            raise ValueError('provided file format: {} cannot be processed'.format(file.__class__))
        # file must be opened in a mode
        if fh.mode != 'a':
            log.info("to save result data, file operating mode need to be 'a'")
            fh.close()
            fh = tl.open_file(file, mode='a')
        try:
            file = fh.filename
            data_node = fh.root
        except tl.exceptions.NoSuchNodeError:
            data_node = fh
            file = fh._v_file.filename
            fh = fh._v_file
        if self.general_opts['node']:
            data_node = fh.get_node('/' + self.general_opts['node'])

        self.file = file
        self.data_node = data_node
        self.file_handle = fh

    def _get_experiment_info(self):
        """
        read experiment settings from the loaded file
        """
        if self.data_node:
            # check number of trials and conditions in the data
            # TODO: condition sequence might be saved as an array as well, if it is too large
            try:
                self.cond_indtrial = np.array(getattr(self.data_node, '_v_attrs')[self.general_opts['cond']])
            except (KeyError, AttributeError):
                self.cond_indtrial = np.array(getattr(self.data_node, '_v_attrs')[self.general_opts['cond_alt']])

            self.N_trials = self.cond_indtrial.shape[0]
            self.conditions = np.unique(self.cond_indtrial, axis=0)
            # get number of light channels used
            self.all_channels = getattr(self.data_node, '_v_attrs')[self.general_opts['LED_chs']]
            if self.general_opts['chs_toanalyze'] is not None:
                chs_to_analyze = []
                for ch in self.general_opts['chs_toanalyze']:
                    if ch in self.all_channels:
                        chs_to_analyze.append(ch)
                    else:
                        log.warning("channel: {} is specified to be analyzed, but does not exist in experiment"
                                    .format(ch))
                self.channels = chs_to_analyze
            else:
                self.channels = self.all_channels
            # these two attributes should be consistent across different experiments
            try:
                self.trial_duration = getattr(self.data_node, '_v_attrs')['Exp_Settings_trial_duration']
            except (AttributeError, KeyError):
                self.trial_duration = None
            try:
                self.inter_trial_interval = getattr(self.data_node, '_v_attrs')['Exp_Settings_inter_trial_interval']
            except (AttributeError, KeyError):
                self.inter_trial_interval = None
            # get the experiment mode, i.e. if using single or rolling stimuli
            try:
                self.exp_mode = getattr(self.data_node, '_v_attrs')['Exp_Settings_mode']
            except (AttributeError, KeyError):
                # old experiment result, only with single mode
                self.exp_mode = 'Single'
        else:
            raise RuntimeError('data file has not been loaded')

    def _close_file(self):
        """
        close the file already opened by the instance
        """
        if self.file_handle and self.file_handle.isopen:
            self.file_handle.flush()
            self.file_handle.close()
        if self.result_indtrial_fh and self.result_indtrial_fh.isopen:
            self.result_indtrial_fh.flush()
            self.result_indtrial_fh.close()
        self.reference_image = None

    def _load_history(self):
        """
        load options used for previous analysis results
        Returns:
            None
        """
        self.analysis_history = []
        # check if history record is already there, if not create it
        result_group = h5t.get_or_append_node(self.data_node, self.__result_path)
        # check analysis opts is already an attribute
        if result_group._v_attrs.__contains__(self.__opts_name):
            self.analysis_history = deepcopy(getattr(result_group._v_attrs, self.__opts_name))
        else:
            log.info('No analysis has been done on this dataset')
        self.result_group = result_group

    def _write_history(self):
        """
        write current analysis options to an analysis history record
        analysis history is saved in the analysis result node (__result_path) as an attribute (__opts_name)
        Returns:
            None
        """
        # check if history record is already there, if not create it
        result_group = h5t.get_or_append_node(self.data_node, self.__result_path)
        # check analysis opts is already an attribute
        if not result_group._v_attrs.__contains__(self.__opts_name):
            result_group._v_attrs.__setattr__(self.__opts_name, [])
        opts = self._get_tosave_opts()
        all_opts = result_group._v_attrs.__getattr__(self.__opts_name)
        all_opts.append(opts)
        result_group._v_attrs.__setattr__(self.__opts_name, all_opts)

    def _write_dict_record_astable(self, where, name, data):
        """
        write a dictionary to a pytables.Table node. if the table does not exist, create it
        Args:
            where: pytables.Group, where the table to be written is located
            name: name of the table if need to be created
            data: the dictionary to be written
        Returns:
            None
        """
        # check if it is already exist
        if not where.__contains__(name):
            # create the table to hold temporal info
            opts_dtype = h5t.get_dict_dtype(data)
            tbl = self.file_handle.create_table(where._v_pathname,
                                                name, opts_dtype)
        else:
            tbl = where._f_get_child(name)
        h5t.dict_add_row(tbl, data)

    def _get_full_opts(self):
        """
        return full options as a dicionary
        Returns:
            dict
        """
        return {key: deepcopy(getattr(self, self.__group_mapping[key])) for key in self.__groups}

    def _get_tosave_opts(self):
        """
        return full options as a dictionary, without those fields defined in __nosave_opts
        Returns:
            dict
        """
        opts = dict()
        for key in self.__groups:
            sub_opts = deepcopy(getattr(self, self.__group_mapping[key]))
            for todel_key in self.__nosave_opts:
                sub_opts.pop(todel_key, None)
            opts[key] = sub_opts
        return opts

    @staticmethod
    def _get_timewindow(trial_duration, tm_wd):
        """
        get analysis timing window to to be used to prepare result array
        :param trial_duration: experiment duration specified by experimenter
        :param tm_wd: 2-element list/tuple/array, or float, timing window to be calculated, second
        :return: nparray of length 2, start and end of analysis window

        Note: the analysis window for each trial still need to be calculated for each trial, which is done by function
        deltaF_over_F in image_analysis_tools.py
        """
        if not tm_wd:
            tm_wd = np.array([0, trial_duration])
        else:
            if not isinstance(tm_wd, Sized):
                tm_wd = [tm_wd]
            tm_wd = np.array(tm_wd)
            if len(tm_wd) == 1:
                tm_wd = (0, tm_wd[0])
            elif len(tm_wd) > 2:
                tm_wd = tm_wd[:2]
        return tm_wd


def analyze_dir(dir_path, **kwargs):
    """
    analyze all data files in a folder
    Args:
        dir_path: str or os.path, folder to be analyzed
        **kwargs: dict used to configure the analysis. see ISI_analyze.configure()

    Returns:
        None
    """
    if not os.path.isabs(dir_path):
        dir_path = os.path.join(get_config('DATA_ROOT'), dir_path)
    import re
    data_files = [n for n in os.listdir(dir_path) if re.search(r'.h5$', n) and
                  not re.search(r'result', n)]

    ia = ISI_analyze()
    # configure the analysis
    for k, v in kwargs.items():
        ia.configure(k, **v)
    # analyze each file
    for file in data_files:
        file_path = os.path.join(dir_path, file)
        ia.load_file(file_path)
        ia.analyze()
        ia._close_file()
        plt.close('all')


if __name__ == "__main__":
    import os
    import re

    file_folder = "E:\\temp"
    data_files = [n for n in os.listdir(file_folder) if re.search(r'.h5$', n) and
                  not re.search(r'result', n)]
    # 2nd file is from the new experiment protocol
    f_path = os.path.join(file_folder, data_files[1])

    ia = ISI_analyze()
    ia.load_file(f_path)
