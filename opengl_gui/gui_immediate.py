import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from contextlib import contextmanager

import glfw
import numpy as np
import time
from functools import cache

from opengl_gui.gui_shaders import *
from opengl_gui.gui_helper import load_font

class Shaders():
    def setup(self, shader):
        return shader().compile().generate_uniform_functions()

    def __init__(self):
        self.texture_bgr = self.setup(TextureShaderBGR)
        self.texture_rgba = self.setup(TextureShaderRGBA)
        self.texture_r   = self.setup(TextureShaderR)
        self.portal     = self.setup(PortalShader)
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
            'early': { '_start': None, 'start': None, 'current': None, 'last_click': None, 'dragging': False, 'end': None },
            'normal': { '_start': None, 'start': None, 'current': None, 'last_click': None, 'dragging': False, 'end': None },
            'late': { '_start': None, 'start': None, 'current': None, 'last_click': None, 'dragging': False, 'end': None },
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

        glEnable(GL_STENCIL_TEST)
        self.stencil_depth = 0

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

        self.text_buffer = glGenBuffers(1)
        self.index_buffer = glGenBuffers(1)
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

        pos = (self.mouse_x, self.mouse_y)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.inputs['early'] = { '_start': None, 'start': pos, 'current': pos, 'last_click': self.time, 'dragging': True, 'end': None }
            elif action == glfw.RELEASE:
                self.inputs['early'] = { **self.inputs['early'], 'current': pos, 'dragging': False, 'end': pos }

    def consume_input(self, kind):
        restore = True
        for phase, input in reversed(self.inputs.items()):
            if restore:
                input['_start'] = input['start'] or input['_start']
            else:
                input['_start'] = None
            input['start'] = input['end'] = None
            if kind == phase:
                restore = False

    def resize_event_callback(self, window, width, height) -> None:

        glViewport(0, 0, width, height)

        self.width  = width
        self.height = height 
        self.window_aspect_ratio = width/height

    def key_press_event_callback(self, window, key, scancode, action, mods) -> None:

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, glfw.TRUE)
        if key == glfw.KEY_F11 and action == glfw.PRESS:
            if self.fullscreen:
                glfw.set_window_monitor(window, None, 0, 0, self.window_width//2, self.window_height//2, 0)
            else:
                glfw.set_window_monitor(window, glfw.get_primary_monitor(), 0, 0, self.window_width, self.window_height, 0)
            self.fullscreen = not self.fullscreen

    def use_program(self, program):
        if self.current_program != program:
            self.current_program = program
            glUseProgram(program)

    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.number_of_vertices)

    def poll_events(self):

        self.drag_lock = None
        self.inputs['late'] = { **self.inputs['normal'], '_start': self.inputs['late']['_start'] }
        self.inputs['normal'] = { **self.inputs['early'], '_start': self.inputs['normal']['_start'] }
        self.inputs['early'] = { **self.inputs['early'], 'current': (self.mouse_x, self.mouse_y) }
        for input in self.inputs.values():
            input['_start'], input['start'] = None, input['dragging'] and input['_start'] or input['start']
        self.time = time.time()
        glfw.poll_events()

    def should_window_close(self):
        return glfw.window_should_close(self.window)

    def clear_screen(self):
        glStencilMask(0xFF)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glStencilMask(0x00)

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

    def texture(self, old, data):
        if old == None:
            tex = glGenTextures(1)
            
            if len(data.shape) == 2:
                data = data[:,:,np.newaxis]
            elif len(data.shape) != 3:
                raise ValueError('Unsupported image dimensions')

            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            channels = data.shape[2]
            if channels == 1:
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                flag = GL_RED
                shader = self.shaders.texture_r
            elif channels == 3:
                flag = GL_RGB
                shader = self.shaders.texture_bgr
            elif channels == 4:
                flag = GL_RGBA
                shader = self.shaders.texture_rgba
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, flag, data.shape[1], data.shape[0], 0, flag, GL_UNSIGNED_BYTE, data)
            glGenerateTextureMipmap(tex)
            return shader, tex, data
        else:
            shader, tex, mat = old
            if data is not mat:
                glBindTexture(GL_TEXTURE_2D, tex)
                channels = data.shape[2]
                flag = GL_RGB if channels == 3 else GL_RED if channels == 1 else GL_RGBA
                if data.shape == mat.shape:
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data.shape[1], data.shape[0], flag, GL_UNSIGNED_BYTE, data)
                else:
                    glTexImage2D(GL_TEXTURE_2D, 0, flag, data.shape[1], data.shape[0], 0, flag, GL_UNSIGNED_BYTE, data)
                glGenerateTextureMipmap(tex)
            return shader, tex, data

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

    @cache
    def calculate_positions(self, content, align, scale, text_size):
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

            advance, bearing, size, atlas_info = [font[c][prop] for prop in \
                ['advance', 'bearing', 'size', 'atlas_info']]
            
            chars.append((
                [x + bearing[0], bearing[1] - size[1]] / dims * text_size,
                size / dims * text_size,
                atlas_info,
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
        
        return chars, offset

    def Text(self, content, align='', text_size=1, color=[0,0,0,1], position=[0,0], scale=[1,1]):
        
        chars, offset = self.calculate_positions(content, align, tuple(scale), text_size)

        with self.Container(position=position, scale=scale):
            self.use_program(self.shaders.text.shader_program)
            glBindTexture(GL_TEXTURE_2D, self.font['atlas'])

            vertices = []
            indices = []
            for i, (c_position, c_scale, atlas_info) in enumerate(chars):
                x, y = c_position + offset
                w, h = c_scale
                
                uv_offset, uv_size = atlas_info
                u0, v0 = uv_offset
                u1 = u0 + uv_size[0]
                v1 = v0 + uv_size[1]
                
                vertices.extend([
                    x,     y + h,     u0, v0,  # bottom-left
                    x + w, y + h,     u1, v0,  # bottom-right
                    x + w, y, u1, v1,  # top-right
                    x,     y, u0, v1,  # top-left
                ])
                
                base_index = i * 4
                indices.extend([
                    base_index, base_index + 1, base_index + 2,  # First triangle
                    base_index, base_index + 2, base_index + 3,   # Second triangle
                ])

            vertices = np.array(vertices, dtype=np.float32)
            # glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            # Vertex buffer
            vbo = self.text_buffer
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            #glBindBuffer(GL_ARRAY_BUFFER, vbo)

            # Index buffer
            ibo = self.index_buffer
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32), GL_STATIC_DRAW)

            self.use_program(self.shaders.text.shader_program)
            glBindTexture(GL_TEXTURE_2D, self.font['atlas'])

            # Shared uniforms
            self.shaders.text.uniform_functions['transform'](self.transform)
            self.shaders.text.uniform_functions['color'](color)
            self.shaders.text.uniform_functions['depth'](self.depth)

            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            # TexCoord attribute
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
            glEnableVertexAttribArray(1)

            # Draw all text in one go
            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, ctypes.c_void_p(0))
            #glDeleteBuffers(1, vbo)
            
            # revert
            glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer_array)
            glBindVertexArray(self.VAO)

            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, None)
            glEnableVertexAttribArray(0)

            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
            glEnableVertexAttribArray(1)


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
            
            start, current, end, consume = self.PointerInput(phase='early', time_limit=0.5)
            if start is not None and all([0 <= x <= 1 for x in [*start, *current]]):
                consume()
                return end is not None
        
        return False
    
    def Image(self, data, position=[0,0], scale=[1,1], color=[1,1,1,1], alpha=1, stretch=False, padding=0, zoom=1):
        if type(data) == tuple:
            shader, texture, mat = data
            cleanup = False
        else:
            shader, texture, mat = self.texture(None, data)
            cleanup = True
        shape = mat.shape

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
        if shader == self.shaders.texture_r:
            shader.uniform_functions['color'](color)
            shader.uniform_functions['depth'](self.depth)
            shader.uniform_functions['zoom'](zoom)
        elif shader == self.shaders.texture_bgr:
            shader.uniform_functions['properties']([self.depth, alpha, zoom])
        elif shader == self.shaders.texture_rgba:
            shader.uniform_functions['properties']([self.depth, zoom])
        
        self.draw()
        if cleanup:
            glDeleteTextures(1, texture)

    def PointerInput(self, position=[0,0], scale=[1,1], phase='normal', time_limit=None):

        start, current, last_click, end = [self.inputs[phase][k] for k in ('start', 'current', 'last_click', 'end')]

        inv = np.linalg.inv(self.derive_transform(self.transform, position, scale))
        tcurrent = current and (inv @ [*current, 1])[:2]
        tstart, tend = None, None
        
        if start and (time_limit is None or self.time - last_click < time_limit):
            tstart = (inv @ [*start, 1])[:2]
            if end:
                tend = (inv @ [*end, 1])[:2]

        return tstart, tcurrent, tend, lambda: self.consume_input(phase)


    @contextmanager
    def Drawer(self, value, scale, side='right', offset_edge=0, offset_content=0, lip_thickness=0.1, color=[0,0,0,0], alpha=1, depth=0):
        
        prop = 0 if side == 'right' or side == 'left' else 1 # 0=x, 1=y
        
        # Position drawer according to value
        movement_range = scale[prop] - offset_content
        if side == 'top':
            pos = [offset_edge, offset_content - scale[1] + value * movement_range]
        elif side == 'right':
            pos = [1 - offset_content - value * movement_range, offset_edge]
        elif side == 'bottom':
            pos = [offset_edge, 1 - offset_content - value * movement_range]
        elif side == 'left':
            pos = [offset_content - scale[0] + value * movement_range, offset_edge]

        grabbed = False
        # Process input
        start, cur1, _, _ = self.PointerInput(phase='late')
        _, cur2, end, consume = self.PointerInput()
        if start is not None and end is None:
            if offset_edge <= cur2[1-prop] <= offset_edge + scale[1-prop]:
                if side == 'left' or side == 'top':
                    if pos[prop] <= cur2[prop] <= pos[prop] + scale[prop] + lip_thickness:
                        grabbed = True
                        value += (cur2[prop] - cur1[prop])/movement_range
                        self.drag_lock = 'drawer'
                        consume()
                else:
                    if pos[prop] - lip_thickness <= cur2[prop] <= pos[prop] + scale[prop]:
                        grabbed = True
                        value -= (cur2[prop] - cur1[prop])/movement_range
                        self.drag_lock = 'drawer'
                        consume()

        if not grabbed:
            value += ((1-value)*0.2 + 0.02) if value > 0.5 else -(value*0.2 + 0.02)

        with self.Container(position=pos, scale=scale, color=color, alpha=alpha, depth=depth):
            yield np.clip(value, 0, 1), grabbed
    
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
            if start is not None and all([0 <= x <= 1 for x in start]) and self.drag_lock is None:
                consume()
                if end is None:
                    return True, lower + (upper - lower) * np.clip((current[0]-sx)/(1-2*sx), 0, 1)
                else:
                    return False, value
            
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

    @contextmanager
    def Portal(self, show_through=True, radius=200, thickness=10, position=[0,0], scale=[1,1], color=[0,0,0,1.0], background=[0,0,0,0]):
        shader = self.shaders.portal

        with self.Container(position=position, scale=scale, color=background):
            wpx, hpx = self.query_container_size_px() # for aspect ratio
            # check if whole container is visible
            if radius > wpx/2 and radius > hpx/2:
                yield
                return

            glStencilMask(0xFF) # make stencil buffer writable
            glStencilOp(GL_KEEP, GL_KEEP, GL_INCR) # set behaviour to increment matching pixels
            self.stencil_depth += 1

            # convert units for shader
            radius /= hpx
            thickness /= hpx

            # increment stencil by drawing inner area (part=0) without color
            if show_through:
                self.use_program(shader.shader_program)
                shader.uniform_functions["part"](0)
                shader.uniform_functions["transform"](self.transform)
                shader.uniform_functions["color"]([0,0,0,0])
                shader.uniform_functions["properties"]([self.depth, wpx/hpx, radius, thickness])
                self.draw()

            # render content
            glStencilFunc(GL_EQUAL, self.stencil_depth, 0xFF) # need to be within ALL the masks to be drawn
            glStencilMask(0x00) # make stencil buffer unwritable
            yield
            self.stencil_depth -= 1
            glStencilFunc(GL_LEQUAL, self.stencil_depth, 0xFF)

            # draw ring
            self.use_program(shader.shader_program)
            shader.uniform_functions["part"](1)
            shader.uniform_functions["transform"](self.transform)
            shader.uniform_functions["color"](color)
            shader.uniform_functions["properties"]([self.depth, wpx/hpx, radius, thickness])
            self.draw()

            # clear stencil buffer if this was the root portal
            if self.stencil_depth == 0:
                glStencilMask(0xFF)
                glClear(GL_STENCIL_BUFFER_BIT)
                glStencilMask(0x00)
