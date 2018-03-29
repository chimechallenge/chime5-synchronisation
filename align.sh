#!/usr/bin/env bash
#
# Copyright 2018 University of Sheffield (Jon Barker)
# MIT License (https://opensource.org/licenses/MIT)

# Example of how to use the CHiME-5 alignment tools to produce an alignment 
# for a session of the CHiME-5 data


# The location of the chime5 installation - edit to fit or add a symbolic link
chime5_corpus=./CHiME5

session="S03"     # The session to be aligned
dataset="train"   # The session's dataset. Can be 'train' or 'dev'


##### Shouldn't need to edit anything beyond this point. #####

audio_dir=${chime5_corpus}/audio

aligned_dir="./transcriptions_aligned"
align_data="./align_data"

# Make directories for holding results
[ ! -d "$align_data" ]  && mkdir "$align_data"
[ ! -d "$align_data/first_pass" ] && mkdir "$align_data/first_pass"
[ ! -d "$align_data/refined" ] && mkdir "$align_data/refined"
[ ! -d "$aligned_dir" ] && mkdir "$aligned_dir"

# Perform the course alignment pass (slowest step)
python3 estimate_alignment.py --sessions "$session" "$audio_dir"/"$dataset" "$align_data"/first_pass 

# Peform the alignment refinement pass
python3 estimate_alignment.py --refine "$align_data"/first_pass  --sessions "$session" "$audio_dir"/"$dataset"  "$align_data"/refined

# Apply the alignment to the transcript file
python3 align_transcription.py --sessions "$session"  "$align_data"/refined "$chime5_corpus"/transcriptions/"$dataset" "$aligned_dir"

# View the alignment
python3 view_alignments.py  --sessions "$session" "$align_data"/refined
