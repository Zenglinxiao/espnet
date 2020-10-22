#!/bin/bash

audio_file=$1
segments_file=$2

#for segment in `cat $segments_file`;
#while read line
i=0
while IFS= read -r line; do
  j=$i
  st=`echo $line | cut -d' ' -f3`
  end=`echo $line | cut -d' ' -f4`
  dur=`echo $end - $st | bc -l`
  echo $st $end $dur
  if [ $i -le 9 ]; then
    j=`echo "0${i}"`
  fi
  #echo $j
  ffmpeg -ss ${st} -i $audio_file -t ${dur} ${audio_file::-4}_utt_${j}-${st}_${end}.wav < /dev/null
  #sleep 20
  i=`echo $i +1 | bc -l`
done < $segments_file


#while IFS=' ' read st end; do
#  #st=`echo $line | cut -d' ' -f1`
#  #end=`echo $line | cut -d' ' -f2`
#  dur=`echo $end - $st | bc -l`
#  echo $st $end $dur
#  ffmpeg -ss ${st} -i $audio_file -t ${dur} ${audio_file::-4}_test_${st}_${end}.wav
#  #sleep 20
#done < $segments_file
