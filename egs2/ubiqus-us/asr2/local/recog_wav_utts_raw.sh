#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#           2019 RevComm Inc. (Takekatsu Hiramura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

if [ ! -f path.sh ] || [ ! -f cmd.sh ]; then
    echo "Please change current directory to recipe directory e.g., egs/tedlium2/asr1"
    exit 1
fi

. ./path.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=4
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
verbose=1      # verbose option
nj=1
batch_size=1

# feature configuration
do_delta=false
decode_dir=decode

# SAD related
sad_dir=SAD_model

. utils/parse_options.sh || exit 1;
. ./cmd.sh

wav=$1
download_dir=${decode_dir}/download

set -e
set -u
set -o pipefail

if [ ! -f "${wav}" ]; then
    echo "No such WAV file: ${wav}"
    exit 1
fi

base=$(basename $wav .wav)
decode_dir=${decode_dir}/${base}

asr_model_dir=download/exp/asr_train_asr_transformer_e18_raw_bpe_sp
lm_model_dir=download/exp/lm_train_lm_adam_bpe

start=$(date +%s)

if [ -d ${decode_dir}/${base} ]; then
    rm -r ${decode_dir}/${base}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation and SAD detection"

    mkdir -p ${decode_dir}/data_unsegment
    echo "$base $wav" > ${decode_dir}/data_unsegment/wav.scp
    echo "X $base" > ${decode_dir}/data_unsegment/spk2utt
    echo "$base X" > ${decode_dir}/data_unsegment/utt2spk
    echo "$base X" > ${decode_dir}/data_unsegment/text

    steps/segmentation/detect_speech_activity.sh --nj 1 --cmd run.pl \
      ${decode_dir}/data_unsegment \
      SAD_model \
      ${decode_dir}/mfcc \
      SAD_model \
      ${decode_dir}/data

    cp ${decode_dir}/data_seg/utt2spk ${decode_dir}/data_seg/text
    mv ${decode_dir}/data_seg ${decode_dir}/data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Split audio to utt-level wav files"
    local/split_audio_to_utts.sh $wav ${decode_dir}/data/segments
   
    mkdir -p ${decode_dir}/audio_utts
    mv ${wav::-4}_utt*wav ${decode_dir}/audio_utts
    for audio_utt in ${decode_dir}/audio_utts/*; do
        utt_file=$(basename ${audio_utt})
        echo "${utt_file::-4} ${audio_utt}" >> ${decode_dir}/wav.scp
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 2: Decoding"

    mkdir ${decode_dir}/log
    key_file=${decode_dir}/wav.scp
    split_nj=$(min "${nj}" "$(<${key_file} wc -l)")
    for n in $(seq "${split_nj}"); do
        split_scps+=" ${decode_dir}/log/keys.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    ${decode_cmd} JOB=1:$split_nj ${decode_dir}/log/decode.JOB.log \
	  asr_inference.py \
	  	--output_dir ${decode_dir}/output.JOB \
	  	--data_path_and_name_and_type ${decode_dir}/wav.scp,speech,sound \
	  	--asr_train_config ${asr_model_dir}/config.yaml \
	  	--asr_model_file ${asr_model_dir}/valid.acc.best.pth \
	  	--key_file ${decode_dir}/log/keys.JOB.scp \
	  	--ngpu $ngpu \
	  	--beam_size 5 \
	  	--batch_size $batch_size \
		  --lm_train_config ${lm_model_dir}/config.yaml \
		  --lm_file ${lm_model_dir}/valid.acc.best.pth \
	  	--num_workers $split_nj

    for f in token token_int text score; do
        for i in $(seq ${split_nj}); do
            cat ${decode_dir}/output.${i}/1best_recog/${f}
        done | LC_ALL=C sort -k1 > ${decode_dir}/${f}
    done
fi


end=$(date +%s)
time_elapsed=`echo $end - $start | bc -l`
echo "${time_elapsed} sec elapsed" > ${decode_dir}/time.txt
