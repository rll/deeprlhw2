from .mjviewer import MjViewer
from .mjcore import MjModel
from .mjcore import register_license
import os
from mjconstants import *
import sys

if sys.platform.startswith("darwin"):
    register_license(os.path.join(os.path.dirname(__file__),
                              '../vendor/mujoco_osx/LICENSE_DEEPRL.TXT'))
elif sys.platform.startswith("linux"):
    register_license(os.path.join(os.path.dirname(__file__),
                              '../vendor/mujoco_linux/LICENSE_DEEPRL.TXT'))
elif sys.platform.startswith("win"):
    register_license(os.path.join(os.path.dirname(__file__),
                              '../vendor/mujoco_win/LICENSE_DEEPRL.TXT'))
else:
    raise RuntimeError("unrecognized platform %s"%sys.platform)
