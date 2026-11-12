#!/usr/bin/env python3

# Copyright 2018 University of Sheffield (Jon Barker)
# MIT License (https://opensource.org/licenses/MIT)

import json

CHIME5_JSON = 'chime5.json'  # Name of the CHiME5 json metadata file


def chime_data(sets_to_load=None):
    """Load CHiME corpus data for specified sets eg. sets=['train', 'eval']

    Defaults to all
    """
    if sets_to_load is None:
        sets_to_load = ['train', 'dev']

    with open(CHIME5_JSON) as fh:
        data = json.load(fh)

    data = {k: v for (k, v) in data.items() if v['dataset'] in sets_to_load}

    return data


def time_text_to_float(time_string):
    """Convert tramscript time from text to float format."""
    hours, minutes, seconds = time_string.split(':')
    seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return seconds


def time_float_to_text(time_float):
    """Convert tramscript time from float to text format."""
    hours = int(time_float/3600)
    time_float %= 3600
    minutes = int(time_float/60)
    seconds = time_float % 60
    return f'{hours}:{minutes:02d}:{seconds:05.7f}'


def load_transcript(session, root, convert=False):
    """Load final merged transcripts.

    session: recording session name, e.g. 'S12'
    """
    with open(f'{root}/{session}.json') as f:
                transcript = json.load(f)
    if convert:
        for item in transcript:
            for key in item['start_time']:
                item['start_time'][key] = time_text_to_float(item['start_time'][key])
            for key in item['end_time']:
                item['end_time'][key] = time_text_to_float(item['end_time'][key])
    return transcript


def save_transcript(transcript, session, root, convert=False):
    """Save transcripts to json file."""

    # Need to make a deep copy so time to string conversions only happen locally
    transcript_copy = [element.copy() for element in transcript]

    if convert:
        for item in transcript_copy:
            for key in item['start_time']:
                item['start_time'][key] = time_float_to_text(
                    item['start_time'][key])
            for key in item['end_time']:
                item['end_time'][key] = time_float_to_text(
                    item['end_time'][key])

    with open(f'{root}/{session}.json', 'w') as outfile:
        json.dump(transcript_copy, outfile, indent=4)
