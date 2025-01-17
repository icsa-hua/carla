#View all the possible start positions on the map. 

import argparse
import sys 
import time 
import glob 
import os
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt

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

import carla

def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        help='load a new map, use --list to see available maps')
    argparser.add_argument(
        '-pos', '--positions',
        metavar='P',
        default='all',
        help='Indices of the positions that you want to plot on the map. '
                'The indices must be separated by commas (default = all positions)')
    
    args = argparser.parse_args()
    client = carla.Client(args.host,args.port)
    client.set_timeout(10.0)
    sim_world = client.get_world()
    world_map = sim_world.get_map() 

    topology = world_map.get_topology()
    for segment in topology:
        x1, y1 = segment[0].transform.location.x, segment[0].transform.location.y
        x2, y2 = segment[1].transform.location.x, segment[1].transform.location.y
        plt.plot([-x1, -x2], [y1, y2], marker = 'o')
    plt.show()


if __name__ == '__main__':
    main()

