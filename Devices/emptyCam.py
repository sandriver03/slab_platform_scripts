"""
an empty camera, used to test the experiment logic classes
performs no actual actions, but the same methods as in the real cam can be called
"""
from Core.Setting import DeviceSetting
from Core.Device import Device
from GUI.Viewer_subprocess import ImageViewer
from Ultilities.RingBuffer import RingBuffer

import numpy as np
from traits.api import Instance, Float, Any, Property, Str, CInt, Enum, Bool, List, Int, \
    cached_property, on_trait_change
from pyvcam import pvc
import pyvcam.constants as camConst
import time

import logging
log = logging.getLogger(__name__)


PRIMECAM_EXPOUTMODE_LOOKUP = {'name': 'exp_out_mode','First Row': 0, 'All Rows': 1, 'Any Row': 2}
PRIMECAM_TRIGGERMODE_LOOKUP = {'name': 'exp_mode', 'Internal': 1792, 'Edge': 2304, 'Trigger first': 2048}
PRIMECAM_BINNING_LOOKUP = {'name': 'binning', '1x1': 1, '2x2': 2}
PRIMECAM_CLEARMODE_LOOKUP = {'name': 'clear_mode', 'Never': 0, 'preExp': 1, 'preSeq': 2, 'postSeq': 3,
                             'pre&postSeq': 4, 'preExp&postSeq': 5}
PRIMECAM_EXPTIME_LOOKUP = {'name': 'exp_time'}
PRIMECAM_FRAMETIME_TABLE = {
    'USB3': {
        '2x2': {'All Rows': 20, 'Any Row': 10, 'First Row': 10},
        '1x1': {'All Rows': 42.25, 'Any Row': 32.5, 'First Row': 32.5}
    },
    'PCIe':  {
        '1x1': {'All Rows': 11, 'Any Row': 11, 'First Row': 11},
        '2x2': {'All Rows': 5.5, 'Any Row': 5.5, 'First Row': 5.5}
    }
}
PRIMECAM_FANSPEED_TABLE = {'low': camConst.FAN_SPEED_LOW,
                           'medium': camConst.FAN_SPEED_MEDIUM,
                           'high': camConst.FAN_SPEED_HIGH,
                           'off': camConst.FAN_SPEED_OFF}
# delay from trigger onset to all rows exposure start, und 'All Rows' exposure out, ms
PRIMECAM_ALLROWS_DELAY = 9.95


# parameters require to reset stream
stream_params = {'binning', 'shared_ram_buffer', 'sampling_freq'}


class SimulatedCam:
    """
    an empty class used to simulate camera handler
    """

    def __init__(self):
        self.name = 'simulated cam'
        self.is_open = False
        self.readout_port = 0
        self.speed_table_index = 0
        self.gain = 1

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def get_frameCnt(self):
        return 0, 0., 0


class EmptyCamSetting(DeviceSetting):

    # TODO: be careful about exposure time, when the camera is externally triggered
    # currently the exposure time is not validated when switching trigger/exposure out mode

    device_type = 'EmptyCam'

    control_interval = 1.
    exposure_time = CInt(2, dsec='exposure time, ms', lookup_table=PRIMECAM_EXPTIME_LOOKUP,
                         camera_setting=True,
                         reinit=False)
    binning = Enum('1x1', '2x2', dsec='binning factor of camera sensor',
                   lookup_table=PRIMECAM_BINNING_LOOKUP, camera_setting=True)
    trigger_mode = Enum('Internal', 'Edge', 'Trigger first',
                        dsec='trigger mode; refer to the camera manual for more details',
                        lookup_table=PRIMECAM_TRIGGERMODE_LOOKUP, camera_setting=True,
                        reinit=False)
    exp_out_mode = Enum('First Row', 'All Rows', 'Any Row',
                        dsec='exposure output mode; refer to the camera manual for more details',
                        lookup_table=PRIMECAM_EXPOUTMODE_LOOKUP, camera_setting=True,
                        reinit=False)
    clear_mode = Enum('preExp', 'preSeq', 'postSeq', 'pre&postSeq', 'preExp&postSeq', 'Never',
                      dsec='sensor clearing mode; when the camera sensor should be cleared',
                      lookup_table=PRIMECAM_CLEARMODE_LOOKUP, camera_setting=True,
                      reinit=False)
    buffer_time = Float(5, group='primary', dsec='buffered data length in second')
    software_scaling = CInt(2, group='primary', dsec='additional software scaling when saving data')
    interface = Enum('USB3', 'PCIe', group='primary', dsec='connection interface currently used',
                     editable=False)

    shape = Property(depends_on='binning', group='derived',
                     dsec='shape of the image taken by the camera')
    dtype = Str('uint16', group='derived', dsec='data type of the image. Camera property')

    type = Str('image', group='status', dsec='nature of the data')
    trig_freq = Int(30, group='status', dsec='frame trigger frequency, Hz')
    _max_expo = Property(depends_on=['trig_freq', 'exp_out_mode', 'trigger_mode'],
                         group='status',
                         sec='maximum exposure time under current configuration, ms')

    sampling_freq = Property(depends_on=['exposure_time', 'exp_out_mode', 'binning', 'trigger_mode'], group='derived',
                             dsec='frame per second of the camera, depends on exposure time')
    buffer_length = Property(depends_on=['buffer_time', 'sampling_freq'], group='derived',
                             dsec='buffer length in data points')

    @cached_property
    def _get_shape(self):
        if self.binning is '1x1':
            return tuple((-1, ) + (2048, 2048))
        elif self.binning is '2x2':
            return tuple((-1, ) + (1024, 1024))
        else:
            raise ValueError("Binning format '{}' is not supported".format(self.binning))

    @cached_property
    def _get_sampling_freq(self):
        if self.trigger_mode != 'Edge':
            return 1000/(self.exposure_time +
                         PRIMECAM_FRAMETIME_TABLE[self.interface][self.binning][self.exp_out_mode])
        else:
            if self.binning == '1x1':
                return 18.0
            else:
                return 30.0

    @cached_property
    def _get_buffer_length(self):
        return int(self.sampling_freq * self.buffer_time)

    @cached_property
    def _get__max_expo(self):
        if self.trigger_mode == 'Internal':
            # TODO: check specs for the maximum exposure time under this condition
            max_expo = 30000.
        else:
            if self.exp_out_mode == 'All Rows':
                # TODO: check if the cam can be triggered after delay + exposure
                max_expo = 1000. / self.trig_freq - PRIMECAM_ALLROWS_DELAY
            else:
                max_expo = 1000. / self.trig_freq - 1
        return max_expo


class EmptyCam(Device):
    cam = Instance(SimulatedCam)
    setting = Instance(EmptyCamSetting, ())
    viewer = Instance(ImageViewer)

    # verbose mode used to debug
    _verbose = False

    # specify the data type and shape of the output stream
    _output_specs = {}

    # if camera live mode is on
    _live_on = Bool(False)
    # buffer to hold snapshot
    buffer_snapshot = Any
    # buffer to hold live stream
    buffer = Instance(RingBuffer)
    # camera buffer
    buffer_cam = Any
    # for now, to detect if a new frame is arrived, check if the buffer has changed;
    # in the future should change to use callbacks
    _frameCnt = Any
    _frameTms = List  # time stamp for each frame
    _preview = Bool   # if the camera is in preview mode
    _frame_header = Any  # indicate if the camera is in preview mode or not
    _stream_param_changed = Bool(False)  # if stream parameters are changed
    _gen_stop_frame = Bool(False)  # if should put a stop frame into the stream
    _empty_frame = Any

    N_snapshot = Int(0)   # number of snapshot taken
    snapshot_buffer_length = Int(5)  # buffer length to hold snapshots

    def _initialize(self, **kargs):
        # initialize pvc module, find and open the camera
        try:
            pvc.init_pvcam()
        except RuntimeError:  # already initialized
            pass
        if not self.cam:
            self.cam = SimulatedCam()
        if not self.cam.is_open:
            self.cam.open()
        # set readout_port, speed_table_index and gain so the camera works normally
        self.cam.readout_port = 0
        self.cam.speed_table_index = 0
        self.cam.gain = 1

        # decide if need to reconfigure output stream
        if set(self._changed_params) & stream_params:
            self._stream_param_changed = True

        # setting up output stream
        if self.output and self._stream_param_changed:
            # close and delete old stream
            names = list(self.output.keys())
            for out_name in names:
                self.remove_stream(out_name, 'output')
            self.output = None
        if not self.output:
            # create output stream
            self.create_output_stream('tcp', monitor=True)

        # setup output specs as well as buffer
        snapshot_specs = {'type': 'image',
                          'shape': self.setting.shape,
                          'sampling_freq': self.setting.sampling_freq,
                          'dtype': self.setting.dtype,
                          }
        camTiming_specs = {'type': 'timing_signal',
                           'shape': (-1, 2),
                           'sampling_freq': self.setting.sampling_freq,
                           'dtype': np.float32,
                           }
        # snapshot does not need to be saved in each trial
        # self._output_specs['snapshot'] = snapshot_specs
        self._output_specs['timing'] = camTiming_specs

        if self.buffer_snapshot is None or self._stream_param_changed:
            self.buffer_snapshot = np.zeros((self.snapshot_buffer_length, ) + self.setting.shape[1:],
                                            dtype=self.setting.dtype)
            if self.setting.buffer_time > 0:
                self.buffer = RingBuffer(shape=(self.setting.buffer_length, ) + self.setting.shape[1:],
                                         dtype=self.setting.dtype, double=False)
        # parameters used to check new frames
        self._frameCnt = 0
        self._frameTms = []
        self._preview = False
        self._frame_header = (b'd', )
        self._stream_param_changed = False
        self._empty_frame = np.zeros(self.setting.shape[1:], self.setting.dtype)

    def _start(self):
        # run the camera with live mode, assuming exposure time is set/ use external trigger
        if not self._live_on:
            if self._verbose:
                print('turning the live mode on')
            # expo_bytes = self.cam.start_live_cb()
            time.sleep(0.05)
            # self.buffer_cam = self.cam.get_live_frame_cb()

        self._frameCnt = self.cam.get_frameCnt()[0]
        self._frameTms = []
        self.change_state(live_on=True)

    def reset_frameInfo(self):
        self._frameCnt = self.cam.get_frameCnt()[0]
        self._frameTms = []

    def _stop(self):
        self.cam.close()
        if self._verbose:
            print('stopping and un-initializing the camera')

    def _pause(self):
        if self._live_on:
            if self._verbose:
                print('turning live mode off')
            # self.cam.stop_live()
            self.change_state(live_on=False)

    def _configure_validation(self, **kwargs):
        # need to make sure exposure time is shorter than maximum allowed
        if 'exposure_time' in kwargs and kwargs['exposure_time'] >= self.setting._max_expo:
            msg = "The new exposure time: {} is longer than the maximum allowed: {}". \
                format(kwargs['exposure_time'], self.setting._max_expo)
            print(msg)
            log.warning(msg)
            # TODO: shall we stop here?
        return kwargs

    def _configure(self):
        # need to apply camera settings through camera property setters or cam.set_param
        # only configure changed camera settings
        # print(self._changed_params)
        for param in set(self._changed_params) & self.setting.traits(camera_setting=True).keys():
            try:
                val = self.setting.trait(param).lookup_table[getattr(self.setting, param)]
            except KeyError:
                val = getattr(self.setting, param)
            # setattr(self.cam, self.setting.trait(param).lookup_table['name'], val)
            log.info('set camera property {} to {}'.format(
                self.setting.trait(param).lookup_table['name'], val))
            if self._verbose:
                print('set camera property {} to {}'.format(
                    self.setting.trait(param).lookup_table['name'], val))

    def set_fan_speed(self, speed):
        """
        set camera fan speed
        Args:
            speed: string, 'low', 'medium', 'high', 'off'

        Returns:
            None
        """
        if self._verbose:
            print('setting camera fan speed to {}'.format(PRIMECAM_FANSPEED_TABLE[speed]))
        # self.cam.set_param(camConst.PARAM_FAN_SPEED_SETPOINT, PRIMECAM_FANSPEED_TABLE[speed])

    def snapshot(self, mode='replace'):
        """
        take a single image with the camera. only works when 'Ready' or 'Paused'
        Args:
            mode: 'replace' or 'increment', either replace old image or save a new image
        Returns:
            None

        """
        if self.state not in ('Ready', 'Paused'):
            if self.state == 'Running':
                raise Exception('the Camera {} is running. Cannot take snapshot while running.')
            elif self.state == 'Created':
                raise Exception('the Camera {} is not initialized. Please initialize it first'
                                .format(self.getName()))
            elif self.state == 'Stopped':
                raise Exception('the Camera {} is already stopped'.format(self.getName()))
            else:
                raise Exception('Cannot take snapshot in current state: {}'.format(self.state))

        if self._live_on:
            if self._verbose:
                print('taking a frame from camera frame buffer')
            # new_frame = self.buffer_cam.copy()
        else:
            if self._verbose:
                print('taking a new frame using current configured exposure')
            # new_frame = self.cam.get_frame(self.setting.exposure_time).copy()
        if mode == 'replace':
            if self.N_snapshot > 0:
                frame_idx = self.N_snapshot % self.snapshot_buffer_length
            else:
                self.N_snapshot += 1
                frame_idx = 0
        elif mode == 'increment':
            frame_idx = self.N_snapshot % self.snapshot_buffer_length
            self.N_snapshot += 1
        if self._verbose:
            print('current snapshot index is {}'.format(frame_idx))

        # self.buffer_snapshot[frame_idx] = new_frame

    def set_preview(self):
        """
        set the mode to preview. live view from the camera, without saving the frames
        should only work in 'Ready' or 'Paused' state

        Returns:
            None
        """
        if self.state not in ('Ready', 'Paused'):
            raise Exception("the preview only works when the camera is in 'Ready' or 'Paused' state")

        self._preview = True
        self._frame_header = (b'p', )

    def stop_preview(self):
        """
        clear preview mode. now the data will be saved
        Returns:
            None
        """
        self._preview = False
        self._frame_header = (b'd', )

    def toggle_liveview(self):
        print('toggle_liveview not implemented for the simulated cam')

    def generate_stop_frame(self):
        if self.setting.operating_mode != 'subprocess':
            self._gen_stop_frame = True
        else:
            raise NotImplementedError

    def thread_func(self):
        # decide if new frame has arrived
        # new_frame_val = self.buffer_cam[self.__frame_check_idx[0], self.__frame_check_idx[1]]
        frame_Cnt = self.cam.get_frameCnt()
        if self._verbose:
            print('getting frames from camera and send to output')
        if self._gen_stop_frame:
            if self._verbose:
                print('generating an stop frame and send it to output')
            # need this signal to tell the writer that current trial is finished
            self.output[self.name + '_tcp'].send(self._empty_frame, header=(b's',))
            self._gen_stop_frame = False


# test
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

    dcam = EmptyCam()
    # fd.setting.configure_traits()
