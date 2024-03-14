# test the Prime Camera

import numpy as np
from traits.api import Instance, Float, Any, Property, Str, CInt, Enum, Bool, List
from pyvcam import pvc
from pyvcam.camera import Camera
import time

from Core.Setting import DeviceSetting
from Core.Device import Device
from GUI.Viewer import ImageViewer
from Ultilities.RingBuffer import RingBuffer

import logging
log = logging.getLogger(__name__)


PRIMECAM_EXPOUTMODE_LOOKUP = {'name': 'exp_out_mode','First Row': 0, 'All Rows': 1, 'Any Row': 2}
PRIMECAM_TRIGGERMODE_LOOKUP = {'name': 'exp_mode', 'Internal': 1792, 'Edge': 2304, 'Trigger first': 2048}
PRIMECAM_BINNING_LOOKUP = {'name': 'binning', '1x1': 1, '2x2': 2}
PRIMECAM_CLEARMODE_LOOKUP = {'name': 'clear_mode', 'Never': 0, 'preExp': 1, 'preSeq': 2, 'postSeq': 3,
                            'pre%postSeq': 4, 'preExp%postSeq': 5}
PRIMECAM_EXPTIME_LOOKUP = {'name': 'exp_time'}
PRIMECAM_FRAMETIME_TABLE = {
    'USB3': {
              '1x1': {'All Rows': 42, 'Any Row': 32, 'First Row': 32},
              '2x2': {'All Rows': 14.5, 'Any Row': 19.5, 'First Row': 19.5}
            },
    'PCIe':  {
              '1x1': {'All Rows': 11, 'Any Row': 11, 'First Row': 11},
              '2x2': {'All Rows': 5.5, 'Any Row': 5.5, 'First Row': 5.5}
            }
    }

# for the prime camera, once the start_live method is called, seems the parameters cannot be modified; otherwise
# most of time the camera simply stops
# IMPORTANT: when use start_live_cb(), must wait at least 0.05s before calling get_live_frame_cb(); otherwise the buffer
#  will not refresh
class testCamSetting(DeviceSetting):

    device_type = 'PrimeCam'

    control_interval = 0.01
    exposure_time = CInt(5, dsec='exposure time, ms', lookup_table=PRIMECAM_EXPTIME_LOOKUP,
                         camera_setting=True)
    binning = Enum('1x1', '2x2', dsec='binning factor of camera sensor',
                   lookup_table=PRIMECAM_BINNING_LOOKUP, camera_setting=True)
    trigger_mode = Enum('Internal', 'Edge', 'Trigger first',
                        dsec='trigger mode; refer to the camera manual for more details',
                        lookup_table=PRIMECAM_TRIGGERMODE_LOOKUP, camera_setting=True)
    exp_out_mode = Enum('First Row', 'All Rows', 'Any Row',
                        dsec='exposure output mode; refer to the camera manual for more details',
                        lookup_table=PRIMECAM_EXPOUTMODE_LOOKUP, camera_setting=True)
    clear_mode = Enum('preExp', 'preSeq', 'postSeq', 'pre%postSeq', 'preExp%postSeq', 'Never',
                      dsec='sensor clearing mode; when the camera sensor should be cleared',
                      lookup_table=PRIMECAM_CLEARMODE_LOOKUP, camera_setting=True)
    buffer_time = Float(5, group='primary', dsec='buffered data length in second')
    interface = Enum('USB3', 'PCIe', group='primary', dsec='connection interface currently used',
                       editable=False)

    image_shape = Property(depends_on='binning', group='derived',
                           dsec='shape of the image taken by the camera')
    image_dtype = Str('uint16', group='derived', dsec='data type of the image. Camera property')
    frame_rate = Property(depends_on=['exposure_time', 'exp_out_mode', 'binning'], group='derived',
                          dsec='frame per second of the camera, depends on exposure time')
    buffer_length = Property(depends_on=['buffer_time', 'frame_rate'], group='derived',
                             dsec='buffer length in data points')

    def _get_image_shape(self):
        if self.binning is '1x1':
            return tuple([2048, 2048])
        elif self.binning is '2x2':
            return tuple([1024, 1024])
        else:
            raise ValueError("Binning format '{}' is not supported".format(self.binning))

    def _get_frame_rate(self):
        return 1000/(self.exposure_time + PRIMECAM_FRAMETIME_TABLE[self.interface][self.binning][self.exp_out_mode])

    def _get_buffer_length(self):
        return int(self.frame_rate * self.buffer_time)


class testCam(Device):
    cam = Instance(Camera)
    setting = Instance(testCamSetting, ())
    #t0 = Float(0)
    #t1 = Float(0)
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
    _frameTms = List  # time stamp for each frame

    def _initialize(self, **kargs):
        # initialize pvc module, find and open the camera
        try:
            pvc.init_pvcam()
        except RuntimeError:  # already initialized
            pass
        if not self.cam:
            # here assume only one camera is connected to the system
            cam = next(Camera.detect_camera())
            # need to make sure only one instance of the camera exist
            for o_cam in cam.get_instances():
                if cam.name == o_cam.name:
                    # already exist, should use the old one and delete newly created
                    del cam
                    cam = o_cam
                    break
            self.cam = cam
        if not self.cam.is_open:
            self.cam.open()
        # set readout_port, speed_table_index and gain so the camera works normally
        self.cam.readout_port = 0
        self.cam.speed_table_index = 0
        self.cam.gain = 1
        # setup viewer
        if not self.viewer:
            self.viewer = ImageViewer()
        self.viewer.configure(image_shape=np.array(self.setting.image_shape),
                              data_type=self.setting.image_dtype,
                              image_name=self.setting.device_name+'_liveview')
        # setup output specs as well as buffer
        self._output_specs = {'type': 'image', 'shape': self.setting.image_shape,
                              'sampling_freq': self.setting.frame_rate,
                              'dtype': self.setting.image_dtype}
        self.buffer_snapshot = np.zeros(self.setting.image_shape, dtype=self.setting.image_dtype)
        self.buffer = RingBuffer(shape=(self.setting.buffer_length, ) + self.setting.image_shape,
                                 dtype=self.setting.image_dtype)
        # parameters used to check new frames
        self._frameCnt = 0
        self._frameTms = []

    def _start(self):
        # run the camera with live mode, assuming exposure time is set/ use external trigger
        if not self._live_on:
            expo_bytes = self.cam.start_live_cb()
            time.sleep(0.05)
            self.buffer_cam = self.cam.get_live_frame_cb()

        self._frameCnt = self.cam.get_frameCnt()[0]
        #self.viewer.data_stream = self.buffer
        #self.viewer.data_stream_type = 'ringbuffer'
        self._frameTms = []
        self.change_state(live_on=True)

    def _stop(self):
        self.cam.close()
        pvc.uninit_pvcam()

    def _pause(self):
        # if trigger mode is not internal, then pause should be handled by trigger signal
        if self.setting.trigger_mode == 'Internal':
            self.cam.stop_live()
        else:
            log.info("camera trigger mode is {}; pausing the *actual* camera should be handled by hardware "
                     "trigger signal".format(self.setting.trigger_mode))
        self.change_state(live_on=False)

    def _configure(self):
        # need to apply camera settings through camera property setters or cam.set_param
        # only configure changed camera settings
        for param in set(self._changed_params) & self.setting.traits(camera_setting=True).keys():
            try:
                val = self.setting.trait(param).lookup_table[getattr(self.setting, param)]
            except KeyError:
                val = getattr(self.setting, param)
            setattr(self.cam, self.setting.trait(param).lookup_table['name'], val)
            log.info('set camera property {} to {}'.format(
                self.setting.trait(param).lookup_table['name'], val))

    def snapshot(self):
        """
        take a single image with the camera. only works when 'Ready' or 'Paused'
        Returns:

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
            self.buffer_snapshot = self.buffer_cam.copy()
        else:
            self.buffer_snapshot = self.cam.get_frame(self.setting.exposure_time).copy()

    def preview(self):
        """
        live view from the camera, without saving the frames
        should only work in 'Ready' or 'Paused' state

        Returns:
            None
        """
        if self.state not in ('Ready', 'Paused'):
            raise Exception("the preview only works when the camera is in 'Ready' or 'Paused' state")

        if not self._live_on:
            # setup camera to live mode
            expo_bytes = self.cam.start_live_cb()
            time.sleep(0.05)
            self.buffer_cam = self.cam.get_live_frame_cb()

            self.change_state(live_on=True)

        self.viewer.data_stream = self.buffer_cam
        if self.viewer.state is not 'Running':
            self.viewer.start()

    def stop_preview(self):
        self.cam.stop_live()
        self.viewer.pause()
        self.change_state(live_on=False)

    def toggle_liveview(self):
        if self.viewer.state is 'Running':
            self.viewer.pause()
        elif self.viewer.state in ['Ready', 'Paused']:
            self.viewer.start()
        else:
            log.warning("live view not available in current viewer state: {}"
                        .format(self.viewer.state))

    def thread_func(self):
        # decide if new frame has arrived
        # new_frame_val = self.buffer_cam[self.__frame_check_idx[0], self.__frame_check_idx[1]]
        frame_Cnt = self.cam.get_frameCnt()
        if frame_Cnt[0] != self._frameCnt:
            # new frame arrived
            self.buffer.write(self.cam.get_live_frame_cb())
            self._frameCnt = frame_Cnt[0]
            self._frameTms.append((frame_Cnt[0], frame_Cnt[2]))
            if frame_Cnt[0] - self._frameCnt > 1:
                log.warning("{} frame(s) skipped".format(frame_Cnt[0] - self._frameCnt))


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

    dcam = testCam()
    #fd.setting.configure_traits()
