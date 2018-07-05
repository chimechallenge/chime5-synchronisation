# CHiME-5 Array Synchronisation Baseline

## About

This repository contains the code used to obtain the baseline device synchronisation for the CHiME-5 challenge. See the [challenge website](http://spandh.dcs.shef.ac.uk/chime_challenge/) for further details.

CHiME-5 employs audio recording made simultaneously by a number of different devices. The start of each device recording has been synchronised by aligning the onset of a synchronisation tone played at the beginning of each recording session. However, the signals can become progressively out of synch due to a combination of clock drift (on all devices) and occasional frame-dropping on the Kinects. 

This misalignment problem is solved by performing an analysis to align the signal and then using this alignment to provide device-dependent utterance start and end times provided in the transcripts.

This is performed in two steps:

`estimate_alignment.py`: Estimation of a 'recording time' to 'signal delay' mapping between a reference binaural recorder and all other devices. The delay between a pair of channels is estimated at regular intervals throughout the party by locating the peak in a cross-correlation between windows of each signal. Estimation is performed in two passes: first at 10 second intervals, and then at 1 second intervals during periods of rapid change. Estimates are fitted with a linear regression when comparing binaural mics (no frame dropping) and are smoothed with median filter when comparing the binaural mic to Kinect recordings. The alignment is estimated in two passes.

`align_transcription.py`: The transcript files are augmented with the device-dependent utterance timings. The original binaural recorder transcription times are first mapped onto the reference binaural recorder, and then from the reference recorder onto each of the Kinects.

## Installation


1. Edit `align.sh` to set the variable `chime5_corpus` to point to your CHiME-5 installation.

2. Install the python module requirements using

``` 
pip install -r requirements.txt
```

## Usage

A script `align.sh` has been provided to show how the tools have been used to compute the device-dependent times appearing in the CHiME-5 challenge transcription files. The script will process the session S03 but can be easily edited to process other sessions.

The script will run both alignment passes (the initial alignment and the refinement). It then applies the alignment to transcription file to generated the corrected device-dependent timings (which should look like the timings that already in the distributed files).  It will also end by displaying plots of the alignments for each channel. These should look like these

Binaural recorders: [plots/align_S03_P.pdf](plots/align_S03_P.pdf)

Kinect devices: [plots/align_S03_U.pdf](plots/align_S03_U.pdf)

#### Notes

- python3.6 or later is required

- The output transcriptions appear in the directory `transcriptions_align`.

- The script was initially run on the original CHiME-5 audio recordings before any redactions were made. For this reason when it is re-run on the distributed version of the CHiME-5 data it may produced some utterance timings that are different from those appearing in the distributed transcripts.

- It is necessary to run both passes of the alignment process before running align_transcription. Skipping the refinement stage will lead to an error.
