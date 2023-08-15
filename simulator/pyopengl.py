import mujoco as mj
from mujoco.glfw import glfw
from controllers.controller import Controller
from utils import control


class GLFWSimulator:
    def __init__(
            self,
            shadow_hand_xml_filepath: str,
            hand_controller: Controller,
            trajectory_steps: int,
            cam_verbose: bool,
            sim_verbose: bool
    ):
        self._model = mj.MjModel.from_xml_path(filename=shadow_hand_xml_filepath)
        self._hand_controller = hand_controller
        self._trajectory_steps = trajectory_steps
        self._cam_verbose = cam_verbose
        self._sim_verbose = sim_verbose

        self._data = mj.MjData(self._model)
        self._camera = mj.MjvCamera()
        self._options = mj.MjvOption()

        self._window = None
        self._scene = None
        self._context = None

        self._mouse_button_left = False
        self._mouse_button_middle = False
        self._mouse_button_right = False
        self._mouse_x_last = 0
        self._mouse_y_last = 0
        self._terminate_simulation = False

        self._sign = ''
        self._trajectory_iter = iter([])
        self._transition_history = []

        self._init_simulation()
        self._init_controller()
        mj.set_mjcb_control(self._controller_fn)

    @property
    def transition_history(self) -> list:
        return self._transition_history

    def _init_simulation(self):
        self._init_world()
        self._init_callbacks()
        self._init_camera()

    # Initializes world (simulation window)
    def _init_world(self):
        glfw.init()
        self._window = glfw.create_window(1200, 900, 'Shadow Hand Simulation', None, None)
        glfw.make_context_current(window=self._window)
        glfw.swap_interval(interval=1)

        mj.mjv_defaultCamera(cam=self._camera)
        mj.mjv_defaultOption(opt=self._options)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)
        self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

    # Initializes keyboard & mouse callbacks for window navigation utilities
    def _init_callbacks(self):
        glfw.set_key_callback(window=self._window, cbfun=self._keyboard_cb)
        glfw.set_mouse_button_callback(window=self._window, cbfun=self._mouse_button_cb)
        glfw.set_cursor_pos_callback(window=self._window, cbfun=self._mouse_move_cb)
        glfw.set_scroll_callback(window=self._window, cbfun=self._mouse_scroll_cb)

    # Initializes world camera (3D view)
    def _init_camera(self):
        self._camera.azimuth = -180
        self._camera.elevation = -20
        self._camera.distance = 0.6
        self._camera.lookat = [0.37, 0, 0.02]

    # Handles keyboard button events to interact with simulator
    def _keyboard_cb(self, window, key: int, scancode, act: int, mods):
        if act == glfw.PRESS:
            if key == glfw.KEY_BACKSPACE:
                mj.mj_resetData(self._model, self._data)
                mj.mj_forward(self._model, self._data)
            elif key == glfw.KEY_ESCAPE:
                self._terminate_simulation = True
            elif key == glfw.KEY_1:
                self._sign = 'rest'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_2:
                self._sign = 'drop'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_3:
                self._sign = 'middle finger'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_4:
                self._sign = 'no'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_5:
                self._sign = 'yes'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_6:
                self._sign = 'rock'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_7:
                self._sign = 'circle'
                self._hand_controller.set_sign(sign=self._sign)

    # Handles mouse-click events to move/rotate camera
    def _mouse_button_cb(self, window, button, act, mods):
        self._mouse_button_left = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._mouse_button_middle = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._mouse_button_right = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        glfw.get_cursor_pos(window)

    # Handles mouse-move callbacks to navigate camera
    def _mouse_move_cb(self, window, xpos: int, ypos: int):
        dx = xpos - self._mouse_x_last
        dy = ypos - self._mouse_y_last
        self._mouse_x_last = xpos
        self._mouse_y_last = ypos

        if not (self._mouse_button_left or self._mouse_button_middle or self._mouse_button_right):
            return

        width, height = glfw.get_window_size(window=window)
        press_left_shift = glfw.get_key(window=window, key=glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        press_right_shift = glfw.get_key(window=window, key=glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = press_left_shift or press_right_shift

        if self._mouse_button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._mouse_button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            assert self._mouse_button_middle

            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            m=self._model,
            action=action,
            reldx=dx/height,
            reldy=dy/height,
            scn=self._scene,
            cam=self._camera
        )

    # Zooms in/out with the camera inside the simulation world
    def _mouse_scroll_cb(self, window, xoffset: float, yoffset: float):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self._model, action, 0.0, -0.05*yoffset, self._scene, self._camera)

    # Initializes hand controller
    def _init_controller(self):
        self._sign = 'rest'
        self._hand_controller.set_sign(sign=self._sign)

    # Defines controller behavior
    def _controller_fn(self, model: mj.MjModel, data: mj.MjData):
        if self._hand_controller.is_done:
            return

        # Retrieves next trajectory control
        next_ctrl = next(self._trajectory_iter, None)

        # Executes control if ctrl is not None else creates next trajectory between a control transition
        if next_ctrl is None:
            start_ctrl = data.ctrl
            end_ctrl = self._hand_controller.get_next_control(sign=self._sign)

            if end_ctrl is None:
                if self._sim_verbose:
                    print('Sign transitions completed')
            else:
                if self._sim_verbose:
                    print(f'New control transition is set from {start_ctrl} to {end_ctrl}')

                row = [self._sign, self._hand_controller.order - 1, end_ctrl]
                self._transition_history.append(row)

                control_trajectory = control.generate_control_trajectory(
                    start_ctrl=start_ctrl,
                    end_ctrl=end_ctrl,
                    n_steps=self._trajectory_steps
                )
                self._trajectory_iter = iter(control_trajectory)

                if self._sim_verbose:
                    print('New trajectory is computed')
        else:
            data.ctrl = next_ctrl

    # Runs GLFW main loop
    def run(self):
        while not glfw.window_should_close(window=self._window) and not self._terminate_simulation:
            time_prev = self._data.time

            while self._data.time - time_prev < 1.0/60.0:
                mj.mj_step(m=self._model, d=self._data)

            viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
            viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

            if self._cam_verbose:
                print(
                    f'Camera Azimuth = {self._camera.azimuth}, '
                    f'Camera Elevation = {self._camera.elevation}, '
                    f'Camera Distance = {self._camera.distance}, '
                    f'Camera Lookat = {self._camera.lookat}'
                )

            # Update scene and render
            mj.mjv_updateScene(
                self._model,
                self._data,
                self._options,
                None,
                self._camera,
                mj.mjtCatBit.mjCAT_ALL.value,
                self._scene
            )
            mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(window=self._window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
        glfw.terminate()
