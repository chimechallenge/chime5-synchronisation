#!/usr/bin/env python3

# Copyright 2018 University of Sheffield (Jon Barker)
# MIT License (https://opensource.org/licenses/MIT)

import argparse
import pickle
import matplotlib.pylab as plt
import traceback

import transcript_utils as tu

def make_plots(data, device_type, layout, ylim):
    plt.figure()
    plot_keys = [key for key in data.keys() if device_type in key]
    for index, key in enumerate(plot_keys):
        chan_data = data[key]
        times = chan_data['times']
        plt.subplot(*layout, index + 1)
        if 'lag' in chan_data:
            plt.plot(times, chan_data['lag'], '-')
        else:
            plt.plot(times, chan_data['lagL'])
            plt.plot(times, chan_data['lagR'])
        plt.ylim(ylim)
        plt.title(key)
    plt.gcf().tight_layout()


def plot_session(session, path, show_plot=True, save_dir=None):
    """Plot figure for a single session."""
    name = f'{path}/align.{session}.p'
    data = pickle.load(open(name, 'rb'))

    device_type='U'
    make_plots(data, device_type, (2,3), ylim=(0, 1))
    if save_dir is not None:
        plt.savefig(f'{save_dir}/align_{session}_{device_type}.pdf')

    device_type='P'
    make_plots(data, device_type, (1, 3), ylim=(-0.1, 0.1))
    if save_dir is not None:
        plt.savefig(f'{save_dir}/align_{session}_{device_type}.pdf')

    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", help="list of sessions to process (defaults to all)")   
    parser.add_argument("--save", help="path of directory in which to save plots")   
    parser.add_argument("--no_plot", action='store_true', help="suppress display of plot (defaults to false)")   
    parser.add_argument("path", help="path to alignment data")   
    args = parser.parse_args()
    if args.sessions is None:
        sessions = tu.chime_data()
    else:
        sessions = args.sessions.split()

    for session in sessions:
        print(session)
        try:
            plot_session(session, args.path, not args.no_plot, args.save)
        except:
            traceback.print_exc()


if __name__ == '__main__':
    main()
