"""
an adaptation from simple_pyspin, https://github.com/klecknerlab/simple_pyspin

The camera effectively operates with 2 buffers:
    the transport layer buffer (on camera), documented in the camera manual
    the in RAM buffer (on computer), not documented in the camera manual

    check the former with TransferQueue* properties:
        TransferQueueCurrentBlockCount
        TransferQueueMaxBlockCount
        TransferQueueOverflowCount
    check the latter with StreamBuffer* properties:
        StreamBufferHandlingMode
        StreamBufferCountMode
        StreamBufferCountManual
    to change the on Computer buffer size, use StreamBufferCountManual
    to get number of readied images in buffer, use StreamOutputBufferCount
"""
from Core.Setting import DeviceSetting
from Core.Device import Device
from Devices.SimplePySpin import Camera, CameraError
from GUI.Viewer_subprocess import ImageViewer
from Ultilities.RingBuffer import RingBuffer


import numpy as np
import time
from traits.api import Instance, Float, Any, Property, Str, CInt, Enum, Bool, List, Int, \
    cached_property, on_trait_change
import logging
log = logging.getLogger(__name__)

# lookup tables
SENSOR_SIZE = (1440, 1080)  # width, height
BLACKFLY_EXPTIME_LOOKUP = {'name': 'ExposureTime'}
BLACKFLY_FRAMERATE_LOOKUP = {'name': 'AcquisitionFrameRate'}
BLACKFLY_TRIGGERMODE_LOOKUP = {'name': 'TriggerMode'}


# parameters require to reset stream
stream_params = {'binning', 'shared_ram_buffer', 'sampling_freq'}


class BlackFlyCamSetting(DeviceSetting):

    # TODO: be careful about exposure time, when the camera is externally triggered
    # currently the exposure time is not validated when switching trigger/exposure out mode

    device_type = 'BlackFlyCam'

    control_interval = 0.01
    frame_rate = CInt(30, dsec='frame rate, Hz', lookup_table=BLACKFLY_FRAMERATE_LOOKUP,
                      camera_setting=True,
                      reinit=False)
    exposure_time = CInt(1000, dsec='exposure time, micro s', lookup_table=BLACKFLY_EXPTIME_LOOKUP,
                         camera_setting=True,
                         reinit=False)
    binning = Enum('2x2', '1x1', '4x4', dsec='binning factor of camera sensor',
                   camera_setting=True)
    trigger_mode = Enum('On', 'Off',
                        dsec='trigger mode; refer to the camera manual for more details',
                        lookup_table=BLACKFLY_TRIGGERMODE_LOOKUP, camera_setting=True,
                        reinit=False, dynamic=False)
    # TODO: auto Exposure
    auto_mode = Enum('Off', 'On',
                     dsec='set auto exposure mode',
                     camera_setting=True,
                     reinit=False, dynamic=False)
    # TODO: for exposure out, probably need to write it ourselves using the digital control
    # TODO: check the LineSource configuration
    # general idea is first set a line to digital output, and then set `LineSource` to `ExposureActive`
    # we only need two values for the config, 'Off' and 'On'
    expo_out = Enum('Off', 'On',
                    dsec='exposure output mode; refer to the camera manual for more details',
                    camera_setting=True, reinit=False)

    buffer_time = Float(5, group='primary', dsec='buffered data length in second')
    software_scaling = CInt(1, group='primary', dsec='additional software scaling when saving data')
    interface = Enum('USB3', 'PCIe', group='primary', dsec='connection interface currently used',
                     editable=False)

    shape = Property(depends_on='binning', group='derived',
                     dsec='shape of the image taken by the camera')
    dtype = Str('uint16', group='derived', dsec='data type of the image. Camera property')

    type = Str('image', group='status', dsec='nature of the data')
    trig_freq = Int(30, group='status', dsec='frame trigger frequency, Hz')
    _max_expo = Property(depends_on=['trig_freq', 'frame_rate', 'trigger_mode'],
                         group='status',
                         sec='maximum exposure time under current configuration, ms')

    sampling_freq = Property(depends_on=['frame_rate', 'trigger_mode', 'trig_freq'], group='derived',
                             dsec='frame per second of the camera, depends on exposure time')
    buffer_length = Property(depends_on=['buffer_time', 'sampling_freq'], group='derived',
                             dsec='buffer length in data points')

    @cached_property
    def _get_shape(self):
        # image data from the camera is transposed
        if self.binning == '1x1':
            return tuple((-1, ) + SENSOR_SIZE[::-1])
        elif self.binning == '2x2':
            return tuple((-1, ) + (int(SENSOR_SIZE[1] / 2), int(SENSOR_SIZE[0] / 2)))
        elif self.binning == '4x4':
            return tuple((-1, ) + (int(SENSOR_SIZE[1] / 4), int(SENSOR_SIZE[0] / 4)))
        else:
            raise ValueError("Binning format '{}' is not supported".format(self.binning))

    @cached_property
    def _get_sampling_freq(self):
        # TODO: when triggered, it should be in sync with the PrimeCam
        if self.trigger_mode == 'Off':
            return self.frame_rate
        else:
            return self.trig_freq

    @cached_property
    def _get_buffer_length(self):
        return int(self.sampling_freq * self.buffer_time)

    @cached_property
    def _get__max_expo(self):
        if self.trigger_mode == 'Off':
            # TODO: check for the frame reading time
            max_expo = 1000. / self.frame_rate * 1000. - 50
        else:
            max_expo = 1000. / self.trig_freq * 1000 - 50
        return max_expo


class BlackFlyCam(Device):
    cam = Instance(Camera)
    setting = Instance(BlackFlyCamSetting, ())
    viewer = Instance(ImageViewer)

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
    _frameBase = Any  # we cannot change the frameCnt at the C++ side, so need to keep the idx of reference frame
    _frameTms = List  # time stamp for each frame
    _preview = Bool   # if the camera is in preview mode
    _frame_header = Any  # indicate if the camera is in preview mode or not
    _stream_param_changed = Bool(False)  # if stream parameters are changed
    _gen_stop_frame = Bool(False)  # if should put a stop frame into the stream
    _empty_frame = Any

    N_snapshot = Int(0)   # number of snapshot taken
    snapshot_buffer_length = Int(10)  # buffer length to hold snapshots

    def _initialize(self, **kargs):
        # initialize the cam using SimplePySpin
        if not self.cam:
            self.cam = Camera()
        if not self.cam.initialized:
            self.cam.init()

        # default configs for scientific imaging
        self._set_default()

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

        # setup viewer
        if self._stream_param_changed and self.viewer:
            self.viewer.stop()
            self.viewer = None
        if not self.viewer:
            self.viewer = ImageViewer()
            self.viewer.configure(data_shape=self.setting.shape,
                                  data_type=self.setting.dtype,
                                  data_monitor_interval=0.01,
                                  fig_name=self.setting.device_name+'_liveview')
            stream_name = self.name + '_tcp'
            self.viewer._input_params = self.output[stream_name].params.copy()
            self.viewer.run_in_process()

        # setup output specs as well as buffer
        snapshot_specs = {'type': 'image',
                          'shape': self.setting.shape,
                          'sampling_freq': self.setting.sampling_freq,
                          'dtype': self.setting.dtype,
                          }
        camTiming_specs = {'type': 'timing_signal',
                           'shape': (-1, 2),
                           'sampling_freq': self.setting.sampling_freq,
                           'dtype': np.float64,
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
        self._frameBase = 0
        self._frameTms = []
        self._preview = False
        self._frame_header = (b'd', )
        self._stream_param_changed = False
        self._empty_frame = np.zeros(self.setting.shape[1:], self.setting.dtype)

    def _set_default(self):
        """
        default configs for scientific imaging
        :return:
        """
        # sometimes the camera locked in "running" due to improper ending from previous session
        try:
            self.cam.AdcBitDepth = 'Bit12'
        except CameraError:
            self.cam.start()
            time.sleep(0.1)
            self.cam.stop()

        # TODO: some camera settings that should not need to be modified, but also not defaults
        # use AcquisitionMode to select Single/MultiFrame or Continuous; use SingleFrame for snapshot
        self.cam.AcquisitionMode = "Continuous"
        self.cam.GainAuto = 'Off'
        # setting Gain is only possible when GainAuto is Off
        self.cam.Gain = 0
        self.cam.GammaEnable = False
        self.cam.AdcBitDepth = 'Bit12'
        self.cam.PixelFormat = 'Mono16'
        # TODO: important: when changing binning, image Height and Width need to be manually adjusted
        self.cam.BinningSelector = 'All'
        self.cam.BinningVerticalMode = 'Sum'
        self.cam.BinningHorizontalMode = 'Sum'
        self.cam.BinningVertical = 2
        self.cam.BinningHorizontal = 2
        # for now, always manually set frame rate
        self.cam.AcquisitionFrameRateEnable = True
        # use TriggerMode to turn trigger on and off
        # Line0 (Black) is opto-isolated input line
        self.cam.TriggerSource = 'Line0'
        self.cam.TriggerSelector = 'FrameStart'
        self.cam.TriggerActivation = 'RisingEdge'
        # for Exposure, need to configure ExposureMode, ExposureAuto and ExposureTime
        self.cam.ExposureAuto = 'Off'
        self.cam.ExposureMode = 'Timed'
        # exposure time is only writable when ExposureAuto is Off
        # ImageTimestamp or FrameID need to be configured through ChunckModeActive and ChunkSelector
        # need to be set one by one
        # time stamps seem in ns
        self.cam.ChunkModeActive = True
        self.cam.ChunkSelector = 'Timestamp'
        self.cam.ChunkEnable = True
        self.cam.ChunkSelector = 'FrameID'
        self.cam.ChunkEnable = True
        # TODO: we should not use long wait time for the acquisition - very easy to get the camera blocked
        # to check if any frames are lost in the stream buffer, check TransferQueueCurrentBlockCount and
        # TransferQueueOverflowCount
        # each start/stop seems clear the camera buffer when TransferControlMode is in basic or Automatic
        # here we set the camera in RAM buffer to overwrite mode
        self.cam.StreamBufferHandlingMode = 'OldestFirstOverwrite'

    def _start(self):
        # run the camera with live mode, assuming exposure time is set/ use external trigger
        self.cam.start()
        self.change_state(live_on=True)

    def reset_frameInfo(self):
        self._frameBase = self.cam.get_frameCnt()[0]
        self._frameTms = []

    def _stop(self):
        self.cam.close()

    def _pause(self):
        if self._live_on:
            self.cam.stop()
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

    def _change_binning(self, new_binning):
        """
        changing the sensor binning is a bit tricky
        increasing the binning is straight forward; however decreasing it requires some intermediate steps
        :return:
        """
        # first reset the sensor binning to 1x1, and image size to full sensor size
        self.cam.BinningVertical = 1
        self.cam.BinningHorizontal = 1
        self.cam.Width = SENSOR_SIZE[0]
        self.cam.Height = SENSOR_SIZE[1]
        # now change the x and y binning to desired values
        self.cam.BinningHorizontal = int(new_binning[0])
        self.cam.BinningVertical = int(new_binning[-1])

    def _expo_out(self, new_val):
        """
        turn exposure out on or off
        :param new_val:
        :return:
        """
        # prepare line1 for exposure out
        self.cam.LineSelector = 'Line1'
        self.cam.LineMode = 'Output'
        if new_val == 'On':
            self.cam.LineSource = 'ExposureActive'
        else:
            try:
                self.cam.LineSource = 'Off'
            except CameraError:
                log.warning("this output line for Expo Out cannot be turned off")

    def _set_auto_mode(self, switch, auto_expo=True, auto_gain=False):
        """
        turn auto exposure and gain on and off
        :param switch: 'Off' or 'On'
        :param auto_expo:
        :param auto_gain:
        :return:
        """
        if switch == 'Off':
            self.cam.ExposureAuto = 'Off'
            self.cam.GainAuto = 'Off'
        elif switch == 'On':
            if auto_expo:
                self.cam.ExposureAuto = 'Once'
            if auto_gain:
                self.cam.GainAuto = 'Once'
        else:
            raise ValueError('switch value: {} not recognized'.format(switch))

    def _configure(self):
        # need to apply camera settings through camera property setters or cam.set_param
        # only configure changed camera settings
        # print(self._changed_params)
        for param in set(self._changed_params) & self.setting.traits(camera_setting=True).keys():
            if param == 'binning':
                self._change_binning(getattr(self.setting, param))
            elif param == 'auto_mode':
                self._set_auto_mode(getattr(self.setting, param))
            elif param == 'expo_out':
                self._expo_out(getattr(self.setting, param))
            else:
                try:
                    val = self.setting.trait(param).lookup_table[getattr(self.setting, param)]
                except KeyError:
                    val = getattr(self.setting, param)
                setattr(self.cam, self.setting.trait(param).lookup_table['name'], val)
                log.info('set camera property {} to {}'.format(
                    self.setting.trait(param).lookup_table['name'], val))

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
            image = self.cam.get_image()
            new_frame = image.GetNDArray()
        else:
            self.cam.start()
            image = self.cam.get_image()
            new_frame = image.GetNDArray()
            image.Release()
            self.cam.stop()
        if mode == 'replace':
            if self.N_snapshot > 0:
                frame_idx = self.N_snapshot % self.snapshot_buffer_length
            else:
                self.N_snapshot += 1
                frame_idx = 0
        elif mode == 'increment':
            frame_idx = self.N_snapshot % self.snapshot_buffer_length
            self.N_snapshot += 1

        self.buffer_snapshot[frame_idx] = new_frame

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
        if self.viewer.state == 'Running':
            self.viewer.pause()
        elif self.viewer.state in ['Ready', 'Paused', 'Created']:
            self.viewer.start()
        else:
            log.warning("live view not available in current viewer state: {}"
                        .format(self.viewer.state))

    def generate_stop_frame(self):
        if self.setting.operating_mode != 'subprocess':
            self._gen_stop_frame = True
        else:
            raise NotImplementedError

    def thread_func(self):
        # decide if new frame has arrived
        # new_frame_val = self.buffer_cam[self.__frame_check_idx[0], self.__frame_check_idx[1]]
        if self.cam.StreamOutputBufferCount > 0:   # only works if StreamBufferHandlingMode contains Overwrite
            # new frame arrived
            image = self.cam.get_image()
            new_frame = image.GetNDArray()
            cd = image.GetChunkData()
            frame_Cnt = cd.GetFrameID()
            frame_time = cd.GetTimestamp()
            self.output[self.name + '_tcp'].send(new_frame, header=self._frame_header)
            if not self._preview and self.setting.buffer_time > 0:
                self.buffer.write(new_frame)
            if frame_Cnt - self._frameCnt > 1:
                log.warning("{} frame(s) skipped".format(frame_Cnt - self._frameCnt))
            self._frameCnt = frame_Cnt
            self._frameTms.append((frame_Cnt, frame_time))
        if self._gen_stop_frame:
            self.output[self.name + '_tcp'].send(self._empty_frame, header=(b's',))
            self._gen_stop_frame = False


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

    cam = BlackFlyCam()


