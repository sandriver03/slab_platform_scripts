from Devices.SimplePySpin import Camera
import time
import numpy as np
import matplotlib.pyplot as plt


class camblk():
    pass


self = camblk()
cam = Camera()
cam.init()
self.cam = cam

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
self.cam.TriggerSelector = 'FrameStart'
# for now, always manually set frame rate
self.cam.AcquisitionFrameRateEnable = True
# use TriggerMode to turn trigger on and off
# Line0 (Black) is opto-isolated input line
self.cam.TriggerSource = 'Line0'
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

self.cam.ExposureTime = 1000

imgs = []
metas = []
cam.start()
t0 = time.time()
fc = 0
while fc < 120:
    # if cam.TransferQueueCurrentBlockCount > 0:
    image = cam.get_image()
    img = image.GetNDArray()
    cd = image.GetChunkData()
    imgs.append(img)
    cam.TimestampLatch()
    metas.append([cd.GetFrameID(), cd.GetTimestamp(),
                  cam.TransferQueueCurrentBlockCount, cam.TransferQueueOverflowCount,
                  cam.TimestampLatchValue])
    image.Release()
    fc += 1
    if fc == 20:
        print('first pause')
        time.sleep(2)
    if fc == 60:
        print('second pause')
        time.sleep(2)
t1 = time.time()
cam.stop()
