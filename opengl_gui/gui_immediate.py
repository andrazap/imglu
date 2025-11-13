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
            print('Error initializing glfw...')
            exit()

        self.shader_pack = None

        self.inputs = {
            'early': { '_start': None, 'start': None, 'last_click': None, 'dragging': False, 'end': None },
            'normal': { '_start': None, 'start': None, 'last_click': None, 'dragging': False, 'end': None },
        }
        self.drag_lock = None

        self.mouse_x = self.mouse_y = -1

        self.mouse_in_window = 0

        self.frames = 0

        self.time = time.time()

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

        self.window = glfw.create_window(width, height, 'VICOS Demo', glfw.get_primary_monitor() if self.fullscreen else None, None)
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
        self.current_program = None

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
                self.inputs['early'] = { '_start': None, 'start': (self.mouse_x, self.mouse_y), 'last_click': self.time, 'dragging': True, 'end': None }
            elif action == glfw.RELEASE:
                self.inputs['early'] = { **self.inputs['early'], 'dragging': False, 'end': (self.mouse_x, self.mouse_y) }

    def consume_input(self):
        for phase, input in self.inputs.items():
            self.inputs[phase] = { **input, '_start': input['start'] or input['_start'], 'start': None, 'end': None }

    def resize_event_callback(self, window, width, height) -> None:

        glViewport(0, 0, self.width, self.height)

        self.width  = width
        self.height = height 
        self.window_aspect_ratio = width/height

    def key_press_event_callback(self, window, key, scancode, action, mods) -> None:

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, glfw.TRUE)

    def use_program(self, program):
        if self.current_program != program:
            self.current_program = program
            glUseProgram(program)

    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.number_of_vertices)

    def poll_events(self):

        # The lock is is a bit of a hack to make sure 
        if self.inputs['normal']['dragging'] and not self.inputs['early']['dragging']:
            self.drag_lock = None
        i = self.inputs['normal'] = self.inputs['early']
        self.inputs['early'] = { **self.inputs['early'], '_start': None, 'start': i['dragging'] and i['_start'] or i['start'] }
        self.time = time.time()
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

    def create_texture(self, data):
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
        return shader, texture, data.shape
    
    def update_texture(self, texture, data):
        glBindTexture(GL_TEXTURE_2D, texture[1])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data.shape[1], data.shape[0], GL_RGB, GL_UNSIGNED_BYTE, data)
        glGenerateTextureMipmap(texture[1])

    @contextmanager
    def Container(self, position=[0,0], scale=[1,1], color=[0,0,0,0], depth=0, alpha=1.0):
        old_transform = self.transform
        self.transform = self.derive_transform(old_transform, position, scale)
        old_depth = self.depth
        self.depth = old_depth + (1 - old_depth) * depth

        try:
            self.use_program(self.shaders.default.shader_program)
            self.shaders.default.uniform_functions['transform'](self.transform)
            self.shaders.default.uniform_functions['properties']([self.depth, alpha])
            self.shaders.default.uniform_functions['color'](color)
            self.draw()
            yield
        finally:
            self.transform = old_transform
            self.depth = old_depth

    def Text(self, content, align='', text_size=1, color=[0,0,0,1], position=[0,0], scale=[1,1]):
        font = self.font
        space = int(np.ceil((font['size'] >> 6)*0.20))

        init_offset = font[content[0]]['bearing'][0]
        x = -min(init_offset,0)

        chars = []
        dims = np.array([self.width, self.height])

        # adjust text size with container scale so it becomes scale independent
        # note: we multiply the Text element's scale so not everything is in the context manager
        text_size /= np.array([self.transform[0,0]*scale[0], self.transform[1,1]*scale[1]])

        for c in content:
            
            if c == ' ':
                x += space
                continue

            advance, bearing, size, texture = [font[c][prop] for prop in \
                ['advance', 'bearing', 'size', 'texture']]
            
            chars.append((
                [x + bearing[0], bearing[1] - size[1]] / dims * text_size,
                size / dims * text_size,
                texture,
            ))
            x += advance[0] >> 6

        used_space = x*text_size[0] / self.width
        font_height_in_container_units = font['line_height']/dims[1]*-text_size[1]
        # align center by default
        offset = [0.5 - used_space/2, 0.5 + font_height_in_container_units/2]
        # handle vertical alignment
        if 'top' in align:
            offset[1] = font_height_in_container_units
        elif 'bottom' in align:
            offset[1] = 1
        # handle horizontal alignment
        if 'right' in align:
            offset[0] = 1 - used_space
        elif 'left' in align:
            offset[0] = 0

        with self.Container(position=position, scale=scale):
            glUseProgram(self.shaders.text.shader_program)
            for c_position, c_scale, c_texture in chars:
                transform = self.derive_transform(self.transform, c_position + offset, c_scale)
                self.shaders.text.uniform_functions['transform'](transform)
                self.shaders.text.uniform_functions['color'](color)
                self.shaders.text.uniform_functions['depth'](self.depth)
                glBindTexture(GL_TEXTURE_2D, c_texture)
                self.draw()

    def Button(self, text=None, icon=None, align='center', icon_padding=0, text_size=1, position=[0,0], scale=[1,1], background=[0,0,0,0], color=[1,1,1,1]):

        with self.Container(color=background, position=position, scale=scale):

            if icon is not None and text:
                # calculate how wide the icon should be to be square
                wpx, hpx = self.query_container_size_px()
                
                # so we know how much to offset text for
                self.Image(icon, color=color, scale=[hpx/wpx,1], padding=icon_padding)
                self.Text(text, align=align, color=color, position=[1.1*hpx/wpx,0], text_size=text_size)
                
                
            elif icon is not None:
                self.Image(icon, color=color, padding=icon_padding)

            elif text:
                self.Text(text, align=align, color=color, text_size=text_size)
            
            start, current, end, consume = self.PointerInput(phase='early', time_limit=0.1)
            if start is not None and all([0 <= x <= 1 for x in [*start, *current]]):
                consume()
                return end is not None
        
        return False
    
    def Image(self, data, position=[0,0], scale=[1,1], color=[1,1,1,1], alpha=1, stretch=False, padding=0, zoom=1):
        if type(data) == tuple:
            shader, texture, shape = data
            cleanup = False
        else:
            shader, texture, shape = self.create_texture(data)
            cleanup = True

        position = position + scale * np.array([0,1])
        scale = scale * np.array([1,-1])

        self.use_program(shader.shader_program)
        transform = self.derive_transform(self.transform, position, scale)
        
        if not stretch:
            sx,sy = transform[0,0],transform[1,1]
            h,w,_ = shape
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

        glBindTexture(GL_TEXTURE_2D, texture)
        shader.uniform_functions['transform'](transform)
        if 'depth' in shader.uniform_functions:
            shader.uniform_functions['color'](color)
            shader.uniform_functions['depth'](self.depth)
            shader.uniform_functions['zoom'](zoom)
        else:
            shader.uniform_functions['properties']([self.depth, alpha, zoom])
        
        self.draw()
        if cleanup:
            glDeleteTextures(1, texture)

    def PointerInput(self, position=[0,0], scale=[1,1], phase='normal', time_limit=None):

        start, last_click, end = [self.inputs[phase][k] for k in ('start', 'last_click', 'end')]

        inv = np.linalg.inv(self.derive_transform(self.transform, position, scale))
        tcurrent = (inv @ [self.mouse_x, self.mouse_y, 1])[:2]
        tstart, tend = None, None
        
        if start and (time_limit is None or self.time - last_click < time_limit):
            tstart = (inv @ [*start, 1])[:2]
            if end:
                tend = (inv @ [*end, 1])[:2]

        return tstart, tcurrent, tend, self.consume_input


    _drawer_state = {}
    @contextmanager
    def Drawer(self, key, scale, side='right', offset_edge=0, offset_content=0, initially_open=False, lip_thickness=0.1, force_state=None, color=[0,0,0,0], alpha=1, depth=0):
        
        prop = 0 if side == 'right' or side == 'left' else 1
        sign = 1 if side == 'left' or side == 'top' else -1
        
        # Initialize drawer state if not present
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
                    self.drag_lock = 'drawer'
            else:
                if pos[prop] - lip_thickness <= start[prop] <= pos[prop] + scale[prop]:
                    grabbed = self._drawer_state[key][2] = True
                    self.drag_lock = 'drawer'

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
    
    def _grid_helper(self, parts):
        total = 0
        nones = 0
        for part in parts:
            if part is None:
                nones += 1
            else:
                total += part
        if round(total,4) > 1:
            raise ValueError('Grid parts cannot sum up to more than 1')
        if nones > 0:
            nones = (1 - total)/nones
        return [nones if x is None else x for x in parts]

    @contextmanager
    def Grid(self, gap=[0,0], cols=[1], rows=[1], position=[0,0], scale=[1,1], depth=0):
        
        cols = self._grid_helper(cols)
        cumcols = np.cumsum(cols) - cols
        rows = self._grid_helper(rows)
        cumrows = np.cumsum(rows) - rows
        hgap, vgap = gap
        
        def layout(i,j, nogap=False):
            i, ri = (i[0], slice(*i)) if type(i) == tuple else (i, slice(i,i+1))
            j, rj = (j[0], slice(*j)) if type(j) == tuple else (j, slice(j,j+1))
            g = 0 if nogap else 1
            
            return {
                'position': [cumcols[j] + g*hgap/2, cumrows[i] + g*vgap/2],
                'scale': [sum(cols[rj]) - g*hgap, sum(rows[ri]) - g*vgap],
            }
        with self.Container(position=position, scale=scale, depth=depth):
            yield layout

    def Slider(self, value, lower=0, upper=1, thumb_size = 35, track_color=[0.5,0.5,0.5,1], thumb_color=[1,1,1,1], position=[0,0], scale=[1,1]):
        wpx, hpx = self.query_container_size_px()
        yscale = thumb_size / hpx
        centered = position[1] + scale[1] / 2 - yscale / 2
        with self.Container(position=[position[0], centered], scale=[scale[0],yscale]):
            sx = thumb_size / wpx # size of thumb relative to current container
            progress = (value - lower) / (upper - lower) # map value range to 0-1
            px = sx + (progress - sx/2) * (1 - 2*sx) # map so 0-1 maps *only* along the track
            
            # Draw track (thumb sized margin in x, 20% high, centered)
            self.use_program(self.shaders.default.shader_program)
            self.shaders.default.uniform_functions['transform'](self.derive_transform(self.transform, [sx,0.4], [1-2*sx, 0.2]))
            self.shaders.default.uniform_functions['properties']([self.depth, 1])
            self.shaders.default.uniform_functions['color'](track_color)
            self.draw()
            # Draw thumb (fill whole y, position in x)
            self.use_program(self.shaders.circle.shader_program)
            self.shaders.circle.uniform_functions['transform'](self.derive_transform(self.transform, [px, 0], [sx, 1]))
            self.shaders.circle.uniform_functions['properties']([self.depth, 1])
            self.shaders.circle.uniform_functions['color'](thumb_color)
            self.draw()
            
            # Handle dragging
            start, current, end, consume = self.PointerInput(phase='early')
            if start is not None and end is None and all([0 <= x <= 1 for x in start]) and self.drag_lock is None:
                consume()
                return True, lower + (upper - lower) * np.clip((current[0]-sx)/(1-2*sx), 0, 1)
            
            return False, value

    def Toggle(self, state, background=[0,0,0,1], thumb=[1,1,1,1], active=[1,0,0,1], position=[0,0], scale=[1,1]):

        with self.Container(position=position, scale=scale):
            wpx, hpx = self.query_container_size_px()
            colors = [active if state else background]*3 + [thumb]
            shaders_ = [getattr(self.shaders, x) for x in ['default', 'circle', 'circle', 'circle']]
            if wpx > hpx:
                ratio = hpx/wpx
                positions = [[ratio/2, 0], [0, 0], [1-ratio, 0], [1 - 0.9*ratio if state else 0.1*ratio, 0.1]]
                scales = [[1-ratio, 1], [ratio, 1], [ratio, 1], [0.8*ratio, 0.8]]
            else:
                ratio = wpx/hpx
                positions = [[0, ratio/2], [0, 0], [0, 1-ratio], [0.1, 1 - 0.9*ratio if state else 0.1*ratio]]
                scales = [[1, 1-ratio], [1, ratio], [1, ratio], [0.8, 0.8*ratio]]

            for shader, position, scale, color in zip(shaders_, positions, scales, colors):
                self.use_program(shader.shader_program)
                shader.uniform_functions['transform'](self.derive_transform(self.transform, position, scale))
                shader.uniform_functions['properties']([self.depth, 1])
                shader.uniform_functions['color'](color)
                self.draw()
            
            start, _, end, consume = self.PointerInput(phase='early')
            if start is not None and end is not None and all([0 <= x <= 1 for x in [*start, *end]]):
                consume()
                return not state
            
            return state
