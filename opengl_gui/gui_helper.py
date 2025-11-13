from OpenGL.raw.GL.VERSION.GL_1_0 import GL_UNPACK_ALIGNMENT, glPixelStorei
from OpenGL.GL import *
import numpy as np
import freetype
import string

import cairosvg
import io
from PIL import Image

def rasterize_svg(*, path: str, width: int, height: int):
    return np.array(Image.open(io.BytesIO(cairosvg.svg2png(url = path, output_width = width, output_height = height))))[:,:,3].astype(np.uint8)

def HSV_to_RGB(H, S, V):

    C = S*V
    X = C*(1 - abs(((H/60)%2) - 1))
    m = V - C

    L = []
    if H < 60:
        L = [C,X,0]
    elif H < 120:
        L = [X,C,0]
    elif H < 180:
        L = [0,C,X]
    elif H < 240:
        L = [0,X,C]
    elif H < 300:
        L = [X,0,C]
    else:
        L = [C,0,X]

    return [L[0] + m, L[1] + m, L[2] + m, 1.0]

def load_font(path, size = 64*50, dpi = 72):
    font = {}
    face = freetype.Face(path)
    characters = list(string.ascii_lowercase) + \
                list(string.ascii_uppercase) + \
                list(string.digits) + \
                ['Č', 'Š', 'Ž', 'č', 'š', 'ž', ':', '-', '#', '.', ',', '!', '?']

    face.set_char_size(width=0, height=size, hres=0, vres=dpi)
    
    glyphs = []
    for c in characters:
        face.load_char(c)

        bitmap = face.glyph.bitmap
        buffer = np.array(bitmap.buffer, dtype = np.int32).reshape(bitmap.rows, bitmap.width)

        glyphs.append({
            'char' : c, 
            'buffer' : buffer, 
            'size' : (face.glyph.bitmap.width, face.glyph.bitmap.rows),
            'bearing' : (face.glyph.bitmap_left, face.glyph.bitmap_top),
            'advance' : (face.glyph.advance.x, face.glyph.advance.y),
        })
    
    # sort characters by height
    characters = sorted(glyphs, key=lambda c: c['size'][::-1], reverse=True)
    atlas_size = (384, 384)
    atlas = np.zeros(atlas_size)
    occupancy = np.zeros_like(atlas)
    d = 1
    x = y = 0
    font = {}
    gap = 2

    while characters:
        i = 0
        while i < len(characters):
            c = characters[i]
            sx, sy = c['size']
            if 0 <= x + d * sx < atlas_size[1] and not np.any(occupancy[y:y+sy,x:x+sx] if d == 1 else occupancy[y-sy:y,x-sx:x]):
                ox, oy = c['offset'] = (x,y) if d == 1 else (x - sx,y - sy)
                bitmap = c['buffer']
                occupancy[oy:oy + bitmap.shape[0],ox:ox+bitmap.shape[1]] = 1
                atlas[oy:oy + bitmap.shape[0],ox:ox+bitmap.shape[1]] = bitmap
                c['atlas_info'] = (ox/atlas_size[1], oy/atlas_size[0]), (bitmap.shape[1]/atlas_size[1],bitmap.shape[0]/atlas_size[0])
                font[c['char']] = c
                x += d * (bitmap.shape[1] + gap)
                del characters[i]
                continue
            i += 1
        if len(characters) == 0:
            break
        if d == 1:
            x = atlas_size[1]
            y += 2*characters[0]['size'][1] + gap
            d = -1
        else:
            x = 0
            y += gap
            d = 1
                
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, atlas.shape[1], atlas.shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, atlas)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
    font['atlas'] = texture
    font['size'] = size
    font['line_height'] = int(size * face.ascender / face.height) >> 6

    return font

def identity4f():
    return np.matrix([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

class Parameters():

    def __init__(self, *, font, aspect, state):
        self.font   = font
        self.aspect = aspect 
        self.state  = state