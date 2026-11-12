#!/usr/bin/env python3

# Copyright 2018 University of Sheffield (Jon Barker)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
align_transcription.py

Apply the alignments to the transcription file
"""

import pickle
import argparse
import traceback
import numpy as np

import transcript_utils as tu

CHIME_DATA = tu.chime_data()


def correct_time_linear(time_to_correct, linear_fit):
    """Adjust the time using a linear fit of time to lag."""
    corrected = time_to_correct - linear_fit * time_to_correct
    return round_to_sample(corrected)


def correct_time_mapping(time_to_correct, linear_fit, times, lags):
    """Adjust the time using a linear fit + a mapping from time to lag."""
    corrected = np.interp(time_to_correct + linear_fit * time_to_correct,
                          np.array(times) + lags, np.array(times))
    return round_to_sample(corrected)


def round_to_sample(time_in_seconds, sample_rate=16000):
    """

    Args:
        time_in_seconds:
        sample_rate:

    Returns:
        time_in_seconds rounded such that time_in_seconds * sample_rate is an
        integer


    >>> 1 / 16000 - 0.000_062_5
    0.0
    >>> round_to_sample(1)
    1.0
    >>> round_to_sample(1.000_031_25)
    1.0
    >>> round_to_sample(1.000_031_26)
    1.0000625
    >>> round_to_sample(1.000_062_5)
    1.0000625

    """
    return np.round(time_in_seconds * sample_rate) / sample_rate


def align_kinect(kinect, transcription, align_data):
    """Add alignments for a given kinect."""
    last_lag = align_data[kinect]['lag'][-1]
    last_time = align_data[kinect]['times'][-1]
    # Plus 100 in next line just to ensure that there is a lag
    # recorded beyond end of party duration. This is needed to
    # make sure that offset is correctly interpolated for
    # utterances in last few seconds of party given that
    # lags are only estimated every 10 seconds.
    times = [0] + list(align_data[kinect]['times']) + [last_time + 100]
    lags = [0] + list(align_data[kinect]['lag']) + [last_lag]
    for utterance in transcription:
        if 'speaker' not in utterance:  # Skips redacted segments
            continue
        linear_fit = 0 
        pid = utterance['speaker']
        if pid in align_data.keys() and 'linear_fit':
            linear_fit = align_data[pid]['linear_fit'][0]
        utterance['start_time'][kinect] = correct_time_mapping(utterance['start_time']['original'],
                                                               linear_fit, times, lags)
        utterance['end_time'][kinect] = correct_time_mapping(utterance['end_time']['original'],
                                                             linear_fit, times, lags)


def align_participant(pid, transcription, align_data):
    """Add alignments for a given binaural participant recording."""
    for utterance in transcription:
        if 'speaker' not in utterance:  # Skips redacted segments
            continue
        utterance_pid = utterance['speaker']

        to_ref_linear = 0
        from_ref_linear = 0
        if utterance_pid in align_data:
            to_ref_linear = -align_data[utterance_pid]['linear_fit'][0]
        if pid in align_data:
            from_ref_linear = align_data[pid]['linear_fit'][0]
        linear_fit = to_ref_linear + from_ref_linear

        utterance['start_time'][pid] = correct_time_linear(
            utterance['start_time']['original'], linear_fit)
        utterance['end_time'][pid] = correct_time_linear(
            utterance['end_time']['original'], linear_fit)


def align_transcription(session, align_data_path, in_path, out_path):
    """Apply kinect alignments to transcription file."""
    align_data = pickle.load(open(f'{align_data_path}/align.{session}.p', 'rb'))
    transcription = tu.load_transcript(session, in_path, convert=True)

    # Compute alignments for the kinects
    kinects = CHIME_DATA[session]['kinects']
    for kinect in kinects:
        print(kinect)
        align_kinect(kinect, transcription, align_data)

    # Compute alignments for the TASCAM binaural mics
    if CHIME_DATA[session]['dataset'] in ['train', 'dev']:
        pids = CHIME_DATA[session]['pids']
        for pid in pids:
            print(pid)
            align_participant(pid, transcription, align_data)

    tu.save_transcript(transcription, session, out_path, convert=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions",
                        help="list of sessions to process (defaults to all)")
    parser.add_argument("align_path", help="path for the alignment pickle files")
    parser.add_argument("in_path", help="path for the input transcription file")
    parser.add_argument("out_path", help="path for the output transcription files")
    args = parser.parse_args()
    if args.sessions is None:
        sessions = tu.chime_data()
    else:
        sessions = args.sessions.split()

    for session in sessions:
        try:
            print(session)
            align_transcription(session, args.align_path, args.in_path, args.out_path)
        except:
            traceback.print_exc()


if __name__ == '__main__':
    main()
