import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from contextlib import contextmanager

import glfw
import numpy as np
import time

from opengl_gui.gui_shaders import *
from opengl_gui.gui_helper import load_font

class Shaders():
    def setup(self, shader):
        return shader().compile().generate_uniform_functions()

    def __init__(self):
        self.texture_bgr = self.setup(TextureShaderBGR)
        self.texture_r   = self.setup(TextureShaderR)
        self.loading     = self.setup(LoadingShader)
        self.default = self.setup(DefaultShader)
        self.text    = self.setup(TextShader)
        self.circle  = self.setup(CircleShader)

class Gui():

    def __init__(self,
        fullscreen: bool = False,
        width:  int = 1920,
        height: int = 1080,
        font: str = './Metropolis-SemiBold.otf') -> None:
        self.aspect_ratio = width/height

        if not glfw.init():
            print("Error initializing glfw...")
            exit()

        self.shader_pack = None

        self.drag_start = None
        self.drag_end = None

        self.mouse_x = self.mouse_y = -1

        self.mouse_in_window = 0

        self.frames = 0

        self.time_fps = time.time()
        self.time_from_start = time.time()

        self.depth = 0
        self.transform       = np.array([[2,0,-1],
                                          [0,-2,1],
                                          [0,0,1]])
        self.transform_inv = np.linalg.inv(self.transform)

        self.window_values = glfw.get_video_mode(glfw.get_primary_monitor())
        self.window_width  = self.window_values.size.width
        self.window_height = self.window_values.size.height

        self.fullscreen = fullscreen and width == self.window_width and height == self.window_height

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.width  = width  if width  <= 3840 else 3840
        self.height = height if height <= 2160 else 2160

        self.window = glfw.create_window(width, height, "VICOS Demo", glfw.get_primary_monitor() if self.fullscreen else None, None)
        self.window_aspect_ratio = self.width/self.height

        glfw.make_context_current(self.window)
        glfw.set_window_size_limits(self.window, 100, 100, 3850, 2160)
        glfw.set_cursor_pos_callback(self.window,       self.mouse_position_callback)
        glfw.set_cursor_enter_callback(self.window,     self.mouse_enter_callback)
        glfw.set_mouse_button_callback(self.window,     self.mouse_event_callback)

        glfw.set_framebuffer_size_callback(self.window, self.resize_event_callback)

        glfw.set_key_callback(self.window, self.key_press_event_callback)
     
        glfw.swap_interval(1)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GEQUAL)
        glDepthRange(0.0,1.0)
        glClearDepth(0.0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.3, 0.3, 0.3, 1.0)

        ####
        self.shaders = Shaders()

        vertices = np.array(
            [1.0,   0, 1.0, 1.0,
             1.0, 1.0, 1.0, 0.0,
               0, 1.0, 0.0, 0.0,
               0, 1.0, 0.0, 0.0,
               0,   0, 0.0, 1.0,
             1.0,   0, 1.0, 1.0], dtype=np.float32)

        self.vertex_buffer_array = glGenBuffers(1)
        self.number_of_vertices  = len(vertices)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer_array)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer_array)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, None)
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        
        self.font = load_font(font)
    
    def mouse_position_callback(self, window, x_pos, y_pos) -> None:
        self.mouse_x = 2*x_pos/self.width - 1
        self.mouse_y = -2*y_pos/self.height + 1

    def mouse_enter_callback(self, window, entered) -> None:
        self.mouse_in_window = entered

    def mouse_event_callback(self, window, button, action, mods) -> None:

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.drag_start = (self.mouse_x, self.mouse_y)
                self.drag_end = None
            elif action == glfw.RELEASE and self.drag_start: # ignore release if we have no start
                self.drag_end = (self.mouse_x, self.mouse_y)

    def consume_input(self):
        self.drag_start = self.drag_end = None

    def resize_event_callback(self, window, width, height) -> None:

        glViewport(0, 0, self.width, self.height)

        self.width  = width
        self.height = height 
        self.window_aspect_ratio = width/height

    def key_press_event_callback(self, window, key, scancode, action, mods) -> None:

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, glfw.TRUE)

    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.number_of_vertices)

    def poll_events(self):

        glfw.poll_events()

    def should_window_close(self):
        return glfw.window_should_close(self.window)

    def clear_screen(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def swap_buffers(self):

        glfw.swap_buffers(self.window)
    
    def close(self):

        glUseProgram(0)
        glfw.terminate()

    def derive_transform(self, old_transform, position, scale):
        return old_transform @ np.array([[scale[0], 0, position[0]],
                                        [0, scale[1], position[1]],
                                        [0,        0,           1]])

    def query_container_size_px(self):
        return self.transform[0,0]*self.width/2, abs(self.transform[1,1]*self.height/2)

    @contextmanager
    def Container(self, position=[0,0], scale=[1,1], color=[0,0,0,0], depth=0, alpha=1.0):
        old_transform = self.transform
        self.transform = self.derive_transform(old_transform, position, scale)
        old_depth = self.depth
        self.depth = old_depth + (1 - old_depth) * depth
        
        try:    
            glUseProgram(self.shaders.default.shader_program)
            self.shaders.default.uniform_functions["transform"](self.transform)
            self.shaders.default.uniform_functions["properties"]([self.depth, alpha])
            self.shaders.default.uniform_functions["color"](color)
            self.draw()
            yield
        finally:
            self.transform = old_transform
            self.depth = old_depth

    def Text(self, content, align='left', text_size=1, position=[0,0], scale=[1,1], color=[0,0,0,1]):
        font = self.font
        space = int(np.ceil((font["size"] >> 6)*0.20))

        init_offset = font[content[0]]["bearing"][0]
        x = -min(init_offset,0)

        chars = []
        dims = np.array([self.width, self.height]) * scale

        for c in content:
            
            if c == ' ':
                x += space
                continue

            advance, bearing, size, texture = [font[c][prop] for prop in \
                ["advance", "bearing", "size", "texture"]]
            
            chars.append((
                [x + bearing[0], bearing[1] - size[1]] / dims * text_size,
                size / dims * text_size,
                texture,
            ))
            x += advance[0] >> 6

        used_space = x*text_size / self.width
        offset = [0, -(0.5 + font['line_height']*text_size/2/self.width)]
        if align == 'center':
            offset[0] = 0.5 - used_space/2
        elif align == 'right':
            offset[0] = 1 - used_space

        scale = scale * np.array([1,-1])
        with self.Container(position=position, scale=scale):
            without_scale = self.transform * np.array([[0,0,1],[0,0,1], [0,0,0]] + np.eye(3))
            glUseProgram(self.shaders.text.shader_program)
            for c_position, c_scale, c_texture in chars:
                transform = self.derive_transform(without_scale, c_position + offset, c_scale)
                
                self.shaders.text.uniform_functions["transform"](transform)
                self.shaders.text.uniform_functions["color"](color)
                self.shaders.text.uniform_functions["depth"](self.depth)
                glBindTexture(GL_TEXTURE_2D, c_texture)
                self.draw()

    def Button(self, text=None, icon=None, align='center', position=[0,0], scale=[1,1], color=[0,0,0,1]):

        with self.Container(color=color, position=position, scale=scale):

            if icon:
                self.Image(icon)

            if text:
                self.Text(text, align=align)
            
            start, _, end, consume = self.PointerInput()
            if start is not None and end is not None and all([0 <= x <= 1 for x in [*start, *end]]):
                consume()
                return True
        
        return False
    
    def Image(self, data, position=[0,0], scale=[1,1], color=[1,1,1,1], alpha=1, stretch=False, padding=0):
        texture = glGenTextures(1)
        single_channel = False
        
        if len(data.shape) == 2:
            single_channel = True
            data = data[:,:,np.newaxis]
        elif len(data.shape) != 3:
            raise ValueError('Unsupported image dimensions')

        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        if single_channel:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            flag = GL_RED
            shader = self.shaders.texture_r
        else:
            flag = GL_RGB
            shader = self.shaders.texture_bgr
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, flag, data.shape[1], data.shape[0],
                0, flag, GL_UNSIGNED_BYTE, data)
        glGenerateTextureMipmap(texture)
        
        position = position + np.array([0,1])
        scale = scale * np.array([1,-1])

        glUseProgram(shader.shader_program)
        transform = self.derive_transform(self.transform, position, scale)
        
        if not stretch:
            sx,sy = transform[0,0],transform[1,1]
            h,w,_ = data.shape
            aspect_container = (sx*self.width)/(sy*self.height)
            aspect_image = w/h
            diff = aspect_image - aspect_container
            if abs(diff) > 0.05:
                if aspect_image > aspect_container:
                    new_sy = scale[1] * aspect_container / aspect_image
                    ds = scale[1] - new_sy
                    transform = self.derive_transform(self.transform, [position[0], position[1]+ds/2], [scale[0], new_sy])
                else:
                    new_sx = scale[0] * aspect_image / aspect_container
                    ds = scale[0] - new_sx
                    transform = self.derive_transform(self.transform, [position[0]+ds/2, position[1]], [new_sx, scale[1]])

        if padding > 0:
            transform = self.derive_transform(transform, [padding, padding], [1-2*padding, 1-2*padding])
            
        shader.uniform_functions["transform"](transform)
        if single_channel:
            shader.uniform_functions["color"](color)
            shader.uniform_functions["depth"](self.depth)
        else:
            shader.uniform_functions["properties"]([self.depth, alpha])
        
        self.draw()
        glDeleteTextures(1, texture)

    def PointerInput(self, position=[0,0], scale=[1,1]):

        inv = np.linalg.inv(self.derive_transform(self.transform, position, scale))
        tcurrent = (inv @ [self.mouse_x, self.mouse_y, 1])[:2]
        tstart, tend = None, None
        
        if self.drag_start:
            tstart = (inv @ [*self.drag_start, 1])[:2]
            if self.drag_end:
                tend = (inv @ [*self.drag_end, 1])[:2]

        return tstart, tcurrent, tend, self.consume_input


    _drawer_state = {}
    @contextmanager
    def Drawer(self, key, scale, side='right', offset_edge=0, offset_content=0, initially_open=False, lip_thickness=0.1, force_state=None, color=[0,0,0,0], alpha=1, depth=0):
        
        prop = 0 if side == 'right' or side == 'left' else 1
        sign = 1 if side == 'left' or side == 'top' else -1
        
        if key not in self._drawer_state:
            if side == 'top':
                initial_position = [offset_edge, offset_content - scale[1]]
            elif side == 'right':
                initial_position = [1 - offset_content, offset_edge]
            elif side == 'bottom':
                initial_position = [offset_edge, 1 - offset_content]
            elif side == 'left':
                initial_position = [offset_content - scale[0], offset_edge]

            self._drawer_state[key] = [initial_position, initially_open, False]

        if force_state is not None:
            self._drawer_state[key][1] = force_state

        pos, isOpen, grabbed = self._drawer_state[key]
        
        with self.Container(position=pos, scale=scale, color=color, alpha=alpha, depth=depth):
            yield grabbed, isOpen

        start, current, end, consume = self.PointerInput()
        
        if not grabbed and start is not None:
            if not offset_edge <= start[1-prop] <= offset_edge + scale[1-prop]:
                return

            if side == 'left' or side == 'top':
                if pos[prop] <= start[prop] <= pos[prop] + scale[prop] + lip_thickness:
                    grabbed = self._drawer_state[key][2] = True
            else:
                if pos[prop] - lip_thickness <= start[prop] <= pos[prop] + scale[prop]:
                    grabbed = self._drawer_state[key][2] = True

        if grabbed:
            if end is not None or start is None:
                self._drawer_state[key][2] = False
                if side == 'left' or side == 'top':
                    self._drawer_state[key][1] = -pos[prop]  < (scale[prop] - offset_content)/2
                else:
                    self._drawer_state[key][1] = 1 - offset_content - pos[prop] > (scale[prop] - offset_content)/2
                consume()
                return
            
            if force_state is not None:
                return
            
            if isOpen:
                reference = 0 if side == 'left' or side == 'top' else 1 - scale[prop]
                pos[prop] = reference - sign * np.clip(0, -sign * (current[prop] - start[prop]), scale[prop] - offset_content)
            else:
                reference = offset_content - scale[prop] if side == 'left' or side == 'top' else 1 - offset_content
                pos[prop] = reference + sign * np.clip(0, sign * (current[prop] - start[prop]), scale[prop] - offset_content)
        else:
            if isOpen:
                reference = 0 if side == 'left' or side == 'top' else 1 - scale[prop]
            else:
                reference = offset_content - scale[prop] if side == 'left' or side == 'top' else 1 - offset_content
            pos[prop] += (reference - pos[prop]) * 0.2
                