from Core.EventManager import EventManager
import Core.EventManager as EMG
from Ultilities.RingBuffer import RingBuffer
import Stream

from traits.api import HasTraits, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import multiprocessing as mp
import multiprocessing.connection
import time
import threading

def init(ax, ln):
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    print('init')
    return ln,

def update(frame, xdata, ydata, ln):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    print('update')
    return ln,

# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=init, blit=True)
# ani = FuncAnimation(fig, func=update, fargs=[xdata, ydata, ln], frames=np.linspace(0, 2 * np.pi, 128),
#                    init_func=partial(init, ax, ln), blit=True)


def ani_main():
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    return FuncAnimation(fig, func=update, fargs=[xdata, ydata, ln], frames=np.linspace(0, 2 * np.pi, 128),
                  init_func=partial(init, ax, ln), blit=True)


def ani_process(gl):
    ani_obj = ani_main()
    print(ani_obj)
    gl.append(ani_obj)


def plot_process():
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)

    time.sleep(0.01)
    for f in np.linspace(0, 2 * np.pi, 128):
        xdata.append(f)
        ydata.append(np.sin(f))
        ln.set_data(xdata, ydata)
        plt.pause(0.2)
        time.sleep(0.01)


class test_plot_process():

    def __init__(self):
        self.fig = None
        self.ax = None
        self.xdata, self.ydata = [], []
        self.ln = None
        self.x = 0
        self.step = self.step = 2 * np.pi / 256
        self._running = False
        self._live = False
        self.event_queue = None

    def update_figure(self):
        for f in np.linspace(0, 2 * np.pi, 128):
            self.xdata.append(f)
            self.ydata.append(np.sin(f))
            self.ln.set_data(self.xdata, self.ydata)
            plt.pause(0.05)

    def init_fig(self):
        print('generate figure template')
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = plt.plot([], [], 'ro')
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1, 1)
        self.x = 0

    def new_plot(self):
        self.init_fig()
        self.update_figure()
        plt.close()

    def _step_fig(self):
        if not self.fig or not plt.fignum_exists(self.fig.number):
            self.init_fig()
        self._update_fig()

    def _update_fig(self):
        print('add one point')
        self.x += self.step
        self.xdata.append(self.x)
        self.ydata.append(np.sin(self.x))
        self.ln.set_data(self.xdata, self.ydata)
        print('finish adding one point')
        plt.pause(0.05)

    def test_process(self, event_q):
        self.event_queue = event_q
        self._live = True
        while self._live:
            self._read_process_command()
            while self._running:
                # core part of the code; read an input data point and plot it
                self._read_process_command()

    def _read_command(self):
        event = self.event_queue.get()[1]
        event_name = event[0]
        event_para = None
        para_unpack = None
        if event.__len__() > 1:
            event_para = event[1]
        if event.__len__() > 2:
            para_unpack = event[2]
        return event_name, event_para, para_unpack

    def _process_command(self, command):
        if command[0] == 'start':
            print('start')
            self._running = True
        elif command[0] == 'pause':
            print('pause')
            self._running = False
        elif command[0] == 'stop':
            print('stop')
            self._running = False
            self._live = False
        elif command[0] == 'new_fig':
            self.new_plot()
        elif command[0] == 'update':
            self._step_fig()
        else:
            print('unknown command {}'.format(command[0]))

    def _read_process_command(self):
        if not self.event_queue.empty():
            command = self._read_command()
            self._process_command(command)

    def process_func(self):
        self.__init__()
        self.update_figure()
        while True:
            plt.pause(0.1)
        # self.update_figure()


class tpp_traits(HasTraits):
    fig = Any
    ax = Any
    xdata = Any
    ydata = Any
    ln = Any
    x = Any
    step = Any
    event_loop = Any
    event_queue = Any
    thread = Any
    data = Any
    data_x = Any
    input = Any
    _input_params = Any
    process = Any
    _running = Any(False)
    _live = Any(False)
    _thread_running = Any(False)

    def init(self):
        print('init figure')
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = plt.plot([], [], 'ro')
        self.ax.set_xlim(0, 10000)
        self.ax.set_ylim(-5, 5)
        self.x = 0
        self.step = 2 * np.pi / 256
        plt.pause(0.02)

    def update_figure(self):
        for f in np.linspace(0, 2 * np.pi, 128):
            self.xdata.append(f)
            self.ydata.append(np.sin(f))
            self.ln.set_data(self.xdata, self.ydata)
            plt.pause(0.02)

    def process_func(self):
        self.init()
        self.update_figure()
        # self.update_figure()

    def new_plot(self):
        self.init()
        self.update_figure()
        plt.close()

    def looped_process(self, evt_q, evt_q_main):
        # attach an event loop to handle device operations
        if not self.event_loop:
            self.event_loop = EventManager()
        else:
            if self.event_loop.state == 'Running':
                if self.event_loop.setting.operating_mode == 'subprocess':
                    # TODO
                    pass
                else:
                    self.event_loop.pause()
        # make sure the event loop is not running, but configuration is good to be ran
        self.event_loop.configure(operating_mode='thread')
        if self.event_loop.event_queue:
            if hasattr(self.event_loop.event_queue, 'closed') \
                    and not self.event_loop.event_queue.closed:
                self.event_loop.event_queue.close()
        self.event_loop.thread = None
        self.event_loop.lock = None
        self.event_loop.event_queue = None
        # register the handlers for the event loop
        self.event_loop._register_event_loop_controls()
        self.event_loop.subscribe('new_fig', self.new_plot)
        self.event_loop.subscribe('get_state', self._get_state_in_subprocess)
        if isinstance(evt_q, (tuple, list)) and isinstance(evt_q[0], mp.connection.PipeConnection):
            for conn in evt_q:
                EMG.setup_PipeConn(conn)
            subprocess_Q = evt_q[1]
            parent_Q = evt_q[0]
        else:
            parent_Q, subprocess_Q = evt_q
        self.event_queue = evt_q_main
        # setup and start a new process
        self.process = mp.Process(target=self._process_run, args=(subprocess_Q, ))
        self.process.start()
        self.event_loop.lock = threading.Lock()
        self.event_loop.setting.operating_mode = 'subprocess'
        self.event_loop.event_queue = parent_Q

    def _get_state_in_subprocess(self, state_list):
        """
        return state variables when the logic is running in a subprocess
        Args:
            state_list: list or tuple of strings; state variables to quire
        Returns:
            None; the values are put into communication queue
        """
        # does nothing when not running in subprocess
        if not self.event_loop:
            return
        res = dict()
        for var in state_list:
            if var == 'settings':
                res[var] = self.setting.get_parameter_value()
            else:
                try:
                    res[var] = getattr(self, var)
                except AttributeError:
                    try:
                        res[var] = getattr(self, '_' + var)
                    except AttributeError:
                        print('state variable {} does not exist'.format(var))
        try:
            self.event_loop.event_queue.put(('states', res))
        except:
            print(res)

    def _process_run(self, event_q):
        """
        this method runs in the new process; need to start the event loop
        remember: everything here is NOT the same object as in the main process
        """
        self.event_loop._in_subprocess = True
        if isinstance(event_q, mp.connection.PipeConnection):
            EMG.setup_PipeConn(event_q)
        self.event_loop.event_queue = event_q
        # start the event loop
        self.event_loop.configure()
        # self.new_plot()
        self.event_loop.start()
        # self.event_loop.state = 'Running'
        # self.event_loop._thread_running = True
        # self.event_loop.thread_run()
        # ths = threading.enumerate()
        # self.event_loop.event_queue.send([th.__repr__() for th in ths])
        self.data = RingBuffer(shape=(10000, 1), dtype=np.float)
        self.data.write(np.zeros((10000, 1)))
        self.data_x = np.arange(self.data.shape[0])
        self._live = True
        self._setup_thread()
        self.thread.start()
        while self._live:
            self._read_process_command()
            while self._running:
                # core part of the code; read an input data point and plot it
                self._step_fig()
                self._read_process_command()

    def _step_fig(self):
        if not self.fig or not plt.fignum_exists(self.fig.number):
            self.init()
        self._update_fig()

    def _update_fig(self):
        # print('update figure')
        # self.x += self.step
        # self.xdata.append(self.x)
        # self.ydata.append(np.sin(self.x))
        # self.ln.set_data(self.xdata, self.ydata)
        # print('finish updating figure')
        # plt.pause(0.02)
        self.ln.set_data(self.data_x, self.data[-10000:])
        plt.pause(0.1)

    def _reset_fig(self):
        self.x = 0
        self.xdata = []
        self.ydata = []
        self.ln.set_data(self.xdata, self.ydata)
        print('reset fig')

    def test_process(self, event_q):
        self.event_queue = event_q
        self._live = True
        if not self.data:
            self.data = RingBuffer(shape=(10000, 1), dtype=np.float)
            self.data.write(np.zeros((10000, 1)))
            self.data_x = np.arange(self.data.shape[0])
        if self._input_params and not self.input:
            self.connect_datastream(self._input_params)
        self._setup_thread()
        self.thread.start()
        while self._live:
            self._read_process_command()
            while self._running:
                # core part of the code; read an input data point and plot it
                self._step_fig()
                self._read_process_command()

    def run_test_process(self, event_q):
        # close and remove input stream
        if self.input:
            self._input_params = self.input.params
            if self.input.socket:
                self.input.close()
            self.input = None
        self.process = mp.Process(target=self.test_process, args=(event_q, ))
        self.event_queue = event_q
        self.process.start()

    def _setup_thread(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError
        self.thread = threading.Thread(target=self._thread_func)

    def _thread_func(self):
        while self._live:
            while self._thread_running:
                # print('thread running')
                # poll the input stream to receive data
                if self.input.poll():
                    _, d = self.input.recv()
                    self.data.write(d)
                time.sleep(0.1)
            time.sleep(0.5)

    def connect_datastream(self, stream):
        """
        Args:
            stream: an Stream.OutputStream instance
        Returns:
        """
        if not self.input:
            self.input = Stream.InputStream()
        self.input.connect(stream)
        # configure input buffer
        #self.input.set_buffer(size=20000)
        #self.input.buffer.write(np.zeros([10000, 1],
        #                                 dtype=np.float))
        #self.input.buffer_offset = 10000
        if not self.data:
            self.data = RingBuffer(shape=(10000, 1), dtype=np.float)
            self.data.write(np.zeros((10000, 1)))
            # self.input.buffer_offset = 10000
            self.data_x = np.arange(self.data.shape[0])

    def _read_command(self):
        event = self.event_queue.get()[1]
        event_name = event[0]
        event_para = None
        para_unpack = None
        if event.__len__() > 1:
            event_para = event[1]
        if event.__len__() > 2:
            para_unpack = event[2]
        return event_name, event_para, para_unpack

    def _process_command(self, command):
        if command[0] == 'start':
            print('start')
            self._running = True
            self._thread_running = True
        elif command[0] == 'pause':
            print('pause')
            self._running = False
        elif command[0] == 'stop':
            print('stop')
            self._running = False
            self._thread_running = False
            self._live = False
        elif command[0] == 'new_fig':
            self.new_plot()
        elif command[0] == 'update':
            self._step_fig()
        elif command[0] == 'reset_fig':
            self._reset_fig()
        else:
            print('unknown command {}'.format(command[0]))

    def _read_process_command(self):
        if not self.event_queue.empty():
            command = self._read_command()
            self._process_command(command)
        # plt.pause(0.05)
        # time.sleep(0.05)


class test():

    def __init__(self):
        self.running = False
        self.data = np.arange(10)
        self.current_idx = 0
        self.thread = None
        self.control_interval = 0.01
        self._advancing = False

    def thread_run(self):
        while self.running:
            self.thread_func()
            time.sleep(self.control_interval)

    def thread_func(self):
        self._advance_index()
        print(self.data[self.current_idx])

    def _advance_index(self):
        if self._advancing:
            self.current_idx += 1
            if self.current_idx > 9:
                self.current_idx = 0
            self._advancing = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.thread_run)
        self.thread.start()


import Testing.test_evtmgr as tt
import multiprocessing as mp
import importlib
import Stream
import numpy as np

tpt = tt.tpp_traits()
tQ = mp.Queue()
outStream = Stream.OutputStream()
outStream.configure(protocol='tcp', sampling_freq=1000)
tpt.connect_datastream(outStream)
