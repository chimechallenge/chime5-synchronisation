#!/usr/bin/env python3

# Copyright 2018 University of Sheffield (Jon Barker)
# MIT License (https://opensource.org/licenses/MIT)

"""
python3 estimate_alignment.py

Estimate an alignment between the reference binaural recording channel and all other audio 
channels in a session. An alignment is a series of (time, lag) pairs where the 'lag' between 
the two channels is measured at a given 'time' point within the reference signal. Alignment 
are initialially computed at 10 second intervals. In periods where the lag changes rapidly 
the alignment is recomputed at 1 second intervals.

The alignment at a time point is estimated by looking for the offset that maximises a 
cross correlation between the two signals being aligned.

The alignments are stored in pickle files called align.{session}.p. They are then read
and applied to the JSON transcript files by align_transcription.py
"""

import argparse
import cv2
import numpy as np
import pickle
import scipy.signal
import struct
import sys
import traceback
import wave

import transcript_utils as tu

# For Bin-to-bin
BINAURAL_RESOLUTION = 100  # analysis resolution (in seconds)
BINAURAL_SEARCH_DURATION = 0.5  # Max misalignment (in seconds)
BINAURAL_TEMPLATE_DURATION = 20  # Duration of segment that is matched (in seconds)

# # For Bin-to-Kinect
KINECT_RESOLUTION = 10  # analysis resolution (in seconds)
# KINECT_SEARCH_DURATION = 3  # Max misalignment (in seconds)
KINECT_SEARCH_DURATION = 5  # Max misalignment (in seconds)
KINECT_TEMPLATE_DURATION = 20  # Duration of segment that is matched (in seconds)


def wavfile_duration(wavfile):
    """Return the duration of a wav file in seconds."""
    wave_fp = wave.open(wavfile, 'rb')
    sample_rate = wave_fp.getframerate()
    nframes = wave_fp.getnframes()
    return nframes/sample_rate


def readwav(wavfile, duration, start_time=0, channel=0):
    """Read a segment of a wav file.
    
    If stereo then can select channel 0 or channel 1.
    """
    wave_fp = wave.open(wavfile, 'rb')
    sample_rate = wave_fp.getframerate()
    nchannels = wave_fp.getnchannels()
    nsamples = int(duration * sample_rate)
    wave_fp.setpos(int(start_time * sample_rate))
    wavbytes = wave_fp.readframes(nsamples)
    signal = struct.unpack(f'{nsamples * nchannels}h', wavbytes)
    if nchannels == 2:
        signal = signal[channel::2]
    return signal, sample_rate


def find_align(target_wavfile, template_wavfile,
               align_time,  search_duration, template_duration, channel=0, missing=None):
    """Find lag between a pair of signals by correlating a short segment of the template
    signal (from one of the Kinects) against the target signal (the reference binaural recorder)
    
    This is the low level signal matching code called by the align_channel() function. It uses
    cv2.matchTemplate to do the work

    Arguments:
    target_wavfile -- name of the target signal wav file 
    template_wavfile -- name of the template signal wav file 
    align_time -- the time within the signals at which to measure the lag
    search_duration -- the max delay to consider
    template_duration -- length of window to use when computing correlations
    channel -- either use left (0) or right (1) stereo channel of the binaural recorder
    missing -- for dealing with Kinect recordings where a large chunk of data has gone missing.
                 missing = (missing_time, missing_duration) 
                 missing_time - the time at which the missing segment starts
                 missing_duration - the number of seconds of audio missing 
    """

    offset = 0
    if missing is not None:
        missing_time, missing_duration = missing
        if align_time > missing_time:
            offset = missing_duration

    target, _ = readwav(target_wavfile,
                        2.0 * search_duration + template_duration,
                        align_time - search_duration + offset,
                        channel=channel)
    template, sample_rate = readwav(template_wavfile,
                                    template_duration,
                                    align_time)
    result = cv2.matchTemplate(np.array(target, dtype=np.float32),
                               np.array(template, dtype=np.float32),
                               cv2.TM_CCOEFF_NORMED)
    lag = np.argmax(result)/sample_rate - search_duration + offset
    return lag, np.max(result)


def align_channels(ref_fn, target_fn, analysis_times,
                   search_duration, template_duration, missing):
    """Compute the alignment between a pair of channel.
    
    Arguments:
    ref_fn -- name of the reference binaural wav file
    target_fn -- name of the kinect channel wav file
    analysis_times -- a list of time points at which to estimate the delay 
    search_duration -- the max delay to consider
    template_duration -- the length of the window over which to compute correlation
    missing -- for dealing with Kinect recordings where a large chunk of data has gone missing.
                (see find_align())
    """

    lag_score_0 = [find_align(ref_fn, target_fn, analysis_time,
                              search_duration, template_duration,
                              channel=0, missing=missing)
                   for analysis_time in analysis_times]
    lag_score_1 = [find_align(ref_fn, target_fn, analysis_time,
                              search_duration, template_duration,
                              channel=1, missing=missing)
                   for analysis_time in analysis_times]
    (lag_0, score_0) = zip(*lag_score_0)
    (lag_1, score_1) = zip(*lag_score_1)

    return {'times': analysis_times, 'lagL': lag_0,
            'lagR': lag_1, 'scoreL': score_0,
            'scoreR': score_1}


def down_mix_lags(results):
    """Produce single sequence of lag estimate from L and R channel lag estimates."""
    # Choose lag from left of right depending which has best correlation score.
    times = np.array(results['times'])
    lagL = np.array(results['lagL'])
    lagR = np.array(results['lagR'])
    best = np.array(results['scoreL']) > np.array(results['scoreR'])
    return best * lagL + ~best * lagR


def clock_drift_linear_fit(results):
    """Compute best linear time-lag fit for clock drift."""
    times = np.array(results['times'])
    lag = np.array(results['lag'])
    a, _, _, _ = np.linalg.lstsq(times[:, np.newaxis], lag)
    return a


def merge_results(results, new_results):
    """Merge two sets of time-lag results."""
    times = list(results['times']) + new_results['times']
    lags = list(results['lag']) + list(new_results['lag'])
    # sort lags into time order
    results['times'], results['lag'] = zip(*(sorted(zip(times, lags))))


def refine_kinect_lags(results, audiopath, session, target_chan, ref_chan):
    """Refine alignment around big jumps in lag.
    
    The initial alignment is computed at 10 second intervals. If the alignment
    changes by a large amount (>50 ms) during a single 10 second step then the
    alignment is recomputed at a resolution of 1 second intervals.

    Arguments:
    results -- the alignment returned by align_channels()
    audiopath -- the directory containing the audio data
    session -- the name of the session to process (e.g. 'S10')
    target_chan -- the name of the kinect channel to process (e.g. 'U01')
    ref_chan -- the name of the reference binaural recorded (e.g. 'P34')

    Return:
    Note, the function updates the contents of results rather than returns results explicitly
    """
    threshold = 0.05
    search_duration = KINECT_SEARCH_DURATION
    template_duration = KINECT_TEMPLATE_DURATION
    chime_data = tu.chime_data()

    times = np.array(results['times'])
    lag = np.array(results['lag'])
    if len(times) != len(lag):
        # This happens for the one case where a kinect was turned off early
        # and 15 minutes of audio got lost
        print('WARNING: missing lags')
        times = times[:len(lag)]
    dlag = np.diff(lag)
    jump_times = times[1:][dlag > threshold]
    analysis_times = set()

    for time in jump_times:
        analysis_times |= set(list(range(time-10, time+10)))
    analysis_times = list(analysis_times)
    print(len(analysis_times))

    if len(analysis_times) > 0:
        missing = None
        if (('missing' in chime_data[session] and
            target_chan in chime_data[session]['missing'])):
            missing = chime_data[session]['missing'][target_chan]

        ref_fn = f'{audiopath}/{session}_{ref_chan}.wav'
        target_fn = f'{audiopath}/{session}_{target_chan}.CH1.wav'

        new_results = align_channels(ref_fn, target_fn, analysis_times,
                                    search_duration, template_duration, missing=missing)
        new_results['lag'] = down_mix_lags(new_results)
        merge_results(results, new_results)


def align_session(session, audiopath, outpath, chans=None):
    """Align all channels within a given session."""
    chime_data = tu.chime_data()

    # The first binaural recorder is taken as the reference
    ref_chan = chime_data[session]['pids'][0]

    # If chans not specified then use all channels available
    if chans is None:  
        pids = chime_data[session]['pids']
        kinects = chime_data[session]['kinects']
        chans = pids[1:] + kinects

    all_results = dict()  # Empty dictionary for storing results

    for target_chan in chans:
        print(target_chan)

        # For dealing with channels with big missing audio segments
        missing = None
        if (('missing' in chime_data[session] and
             target_chan in chime_data[session]['missing'])):
            missing = chime_data[session]['missing'][target_chan]

        # Parameters for alignment depend on whether target is
        # a binaural mic ('P') or a kinect mic
        if target_chan[0] == 'P':
            search_duration = BINAURAL_SEARCH_DURATION
            template_duration = BINAURAL_TEMPLATE_DURATION
            alignment_resolution = BINAURAL_RESOLUTION
            target_chan_name = target_chan
        else:
            search_duration = KINECT_SEARCH_DURATION
            template_duration = KINECT_TEMPLATE_DURATION
            alignment_resolution = KINECT_RESOLUTION
            target_chan_name = target_chan + '.CH1'

        # Place it try-except block so that can continue
        # if a channel fails. This shouldn't happen unless
        # there is some problem reading the audio data.
        try:
            offset = 0
            if missing is not None:
                _, offset = missing

            ref_fn = f'{audiopath}/{session}_{ref_chan}.wav'
            target_fn = f'{audiopath}/{session}_{target_chan_name}.wav'

            # Will analyse the alignment offset at regular intervals
            session_duration = int(min(wavfile_duration(ref_fn) - offset,
                                   wavfile_duration(target_fn))
                                   - template_duration - search_duration)
            analysis_times = range(alignment_resolution, session_duration, alignment_resolution)

            # Run the alignment code and store results in dictionary
            all_results[target_chan] = \
                align_channels(ref_fn, 
                               target_fn,
                               analysis_times, 
                               search_duration,
                               template_duration, 
                               missing=missing)
        except:
            traceback.print_exc()

    pickle.dump(all_results, open(f'{outpath}/align.{session}.p', "wb"))


def refine_session(session, audiopath, inpath, outpath):
    """Refine alignment of all channels within a given session."""
    chime_data = tu.chime_data()

    ref = chime_data[session]['pids'][0]
    pids = chime_data[session]['pids'][1:]
    kinects = chime_data[session]['kinects']
    all_results = pickle.load(open(f'{inpath}/align.{session}.p', "rb"))
    kinects = sorted(list(set(kinects).intersection(all_results.keys())))
    print(session)

    # Merges results of left and right channel alignments
    for channel in pids + kinects:
        results = all_results[channel]
        lag = down_mix_lags(results)
        results['lag'] = scipy.signal.medfilt(lag, 9)

    # Compute the linear fit for modelling clock drift
    for channel in pids:
        results = all_results[channel]
        results['linear_fit'] = clock_drift_linear_fit(results)

    # Refine kinect alignments - i.e. reanalyse on finer time
    # scale in regions where big jumps in offset occur and
    # apply a bit of smoothing to remove spurious estimates
    for channel in kinects:
        refine_kinect_lags(all_results[channel], audiopath, session=session, target_chan=channel, ref_chan=ref)
        results['lag'] = scipy.signal.medfilt(results['lag'], 7)

    pickle.dump(all_results, open(f'{outpath}/align.{session}.p', "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions",
                        help="list of sessions to process (defaults to all)")
    parser.add_argument("--chans", help="list of channels to process (defaults to all)")
    parser.add_argument("--refine", help="path to output of 1st pass (runs a refinement pass if provided)")

    parser.add_argument("audiopath", help="path to audio data")
    parser.add_argument("outpath", help="path to output alignment data")
    args = parser.parse_args()

    if args.sessions is None:
        sessions = tu.chime_data()
    else:
        sessions = args.sessions.split()

    if args.chans is None:
        chans = None
    else:
        chans = args.chans.split()

    chime_data = tu.chime_data()

    if args.refine:
        # The alignment refinement pass.
        for session in sessions:
            refine_session(session, args.audiopath, args.refine, args.outpath)
    else:
        # The initial alignment pass.
        for session in sessions:
            print(session, chans)
            align_session(session, args.audiopath, args.outpath, chans=chans)


if __name__ == '__main__':
    main()
