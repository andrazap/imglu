import os
import importlib
import traceback
import time
import argparse
import cProfile

parser = argparse.ArgumentParser(
    prog='imglu',
    description='Imglu ui runner')
parser.add_argument('program')
parser.add_argument('--debug', '-d', action='store_true')
args = parser.parse_args()

ui = importlib.import_module(args.program.removesuffix('.py'))
path = './' + args.program
gui, *rest = ui.setup()

last_modified = os.path.getmtime(path)
error = False

while not gui.should_window_close():
    gui.poll_events()
    gui.clear_screen()

    if not error:
        try:
            if args.debug:
                with cProfile.Profile() as pr:
                    ui.draw(gui, *rest)
                pr.print_stats('cumulative')
            else:
                ui.draw(gui, *rest)
        except Exception:
            print(traceback.format_exc())
            error = True
    
    current_modified = os.path.getmtime(path)
    if current_modified != last_modified:
        last_modified = current_modified
        try:
            importlib.reload(ui)
            error = False
        except Exception as e:
            error = True
    
    gui.swap_buffers()

ui.cleanup(gui, *rest)
gui.close()
