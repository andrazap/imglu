from opengl_gui.gui_immediate import Gui
from opengl_gui.gui_helper import rasterize_svg

# Global items will get rerun on save
icon = rasterize_svg(path = "./icon.svg", width = 256, height = 256)
class Color():
    RED = [226.0/255, 61.0/255, 40.0/255.0, 0.75]

class State:
    i = 0
    drawer = 0
    sliderVal = 0
    toggle = False

def setup():
    gui = Gui(
        fullscreen = False,
        width = 960,
        height = 690,
        font="./Metropolis-SemiBold.otf")
    return gui, State()

def draw(gui: Gui, state):
    with gui.Container(position=[0,0], scale=[1,1], color=[0,1,0,1]):
        gui.Text(str(state.sliderVal), align='top')
        with gui.Grid(position=[0,0], scale=[0.5,0.5], rows=[0.3,None], cols=[None, None], gap=[0.1,0.1]) as g:
            held, state.sliderVal = gui.Slider(state.sliderVal, upper=1000, **g(0,(0,2)))
            gui.Button(background=[1,1,0,1], **g(1,0)) and print('2')
            state.toggle = gui.Toggle(state.toggle, **g(1,1))
            
        gui.Text('Wow gee', scale=[0.25,0.25])
        # we only need to set it once, otherwise it is locked in that state
        with gui.Drawer(state.drawer, scale=[0.5,0.5], side='top', lip_thickness=0.2, color=[1,0.5,0,1], depth=1) as (state.drawer, grabbed):
            gui.Image(icon)
        if gui.Button('Close drawer' if state.drawer > 0.5 else 'Open drawer', position=[0, 0.75], scale=[0.25,0.25], color=[0,0,1,1]):
            state.drawer = 0.49 if state.drawer > 0.5 else 0.51
        
        with gui.Container(position=[0.25,0.5], scale=[0.5,0.5], color=[0.5,0.5,0.5,1]):
            gui.Text('Wow gee wilicker', align='right')
            if gui.Button('Yippie!', color=Color.RED):
                print('Pressed', state.i)
                state.i += 1
        with gui.Portal(position=[0.25,0.25], radius=state.sliderVal, show_through=True, scale=[0.75,0.75]):
            gui.Text('Hello stencils', align='top')
            with gui.Portal(position=[0.25, 0.25], show_through=True, scale=[0.75,0.75], background=[0,0,1,1]):
                if gui.Button('Hello', align='top', color=[0,0,1,1], background=[1.0,0.0,0.0,1.0]):
                    print('Am button')