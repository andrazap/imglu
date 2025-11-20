import os
import importlib
import ui
import traceback
import time

gui, *rest = ui.setup()

last_modified = os.path.getmtime('./ui.py')
error = False

while not gui.should_window_close():
    gui.poll_events()
    gui.clear_screen()

    if not error:
        try:
            ui.draw(gui, *rest)
        except Exception:
            print(traceback.format_exc())
            error = True
    
    current_modified = os.path.getmtime('./ui.py')
    if current_modified != last_modified:
        last_modified = current_modified
        try:
            gui._drawer_state = {}
            importlib.reload(ui)
            error = False
        except Exception as e:
            error = True
    
    gui.swap_buffers()
    time.sleep(1/60)
        
gui.close()