from opengl_gui.gui_immediate import Gui
from opengl_gui.gui_helper import rasterize_svg

# Global items will get rerun on save
icon = rasterize_svg(path = "./icon.svg", width = 256, height = 256)
class ViCoS():
    RED = [226.0/255, 61.0/255, 40.0/255.0, 0.75]

def setup(state):
    state.i = 0
    return Gui(
        fullscreen = False,
        width = 960,
        height = 690,
        font="./Metropolis-SemiBold.otf")

def draw(gui: Gui, state):
    with gui.Container(position=[0,0], scale=[1,1], color=[0,1,0,1]):
        gui.Text('Wow gee', align='right')
        with gui.Container(position=[0.25,0.5], scale=[0.5,0.5]):
            gui.Text('Wow gee wilicker')
            if gui.Button('Yippie!', color=ViCoS.RED):
                print('Pressed', state.i)
                state.i += 1
        gui.Image(icon)