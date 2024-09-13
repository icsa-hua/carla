import glob 
import os 
import sys
import warnings

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from agents.navigation.dummy_application.application.app import Application 
from agents.navigation.dummy_application.dummy_utils.GUI.pygame_gui import PyGUI

def main(): 
    # pygame_gui = PyGUI(image_path="C:\carla\PythonAPI\examples\Town_images\Town01.jpg")
    # import pdb; pdb.set_trace()
    app = Application()
    app.get_arguments()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        warnings.warn(f"Error found {e}")