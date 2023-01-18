#!/usr/bin

steps=1
cmd="slurm.pl --quiet"
nj=30
train_set=train
valid_set=cv_all
noise_set=musan_noise
test_sets=test
dump_dir=dump
downsample=8k
upsample=16k
audio_format=wav

. utils/parse_options.sh || exit 1
. ./path.sh

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
  if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
  print $steps;' 2>/dev/null)  || exit 1
  
if [ ! -z "$steps" ]; then
#  echo $steps
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    # echo $index
    declare step$index=1
  done
fi


# downsample 8k and generate 8k audio
if [ ! -z $step01 ]; then
  for dset in "${train_set}" "${valid_set}" ${test_sets} ${noise_set};do

    mkdir -p "${dump_dir}/${downsample}/${dset}"
    cp data_8k/${dset}/{spk2utt,utt2spk,text,utt2IntLabel,utt2dur} ${dump_dir}/${downsample}/${dset}/
    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${cmd}" \
                    --audio-format "${audio_format}" --fs "${downsample}" \
                    "data_8k/${dset}/wav.scp" "${dump_dir}/${downsample}/${dset}"
  done
fi

# add 8k noise and generate 8k noise audio
if [ ! -z $step02 ]; then
    cd Add-Noise
    bash add-noise.sh --steps 2 --src-train ../${dump_dir}/${downsample}/test  --noise_dir ../${dump_dir}/${downsample}/musan_noise   
    cd ..
    for dset in ${test_sets};do
	    for snrs in 5 10 15 20;do
		mv ${dump_dir}/${downsample}/${dset}_${snrs}_snrs/wav.scp  ${dump_dir}/${downsample}/${dset}_${snrs}_snrs/wav.scp.tmp
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${cmd}" \
                    --audio-format "${audio_format}" --fs "${downsample}" \
                     "${dump_dir}/${downsample}/${dset}_${snrs}_snrs/wav.scp.tmp" "${dump_dir}/${downsample}/${dset}_${snrs}_snrs"
	    done
    done   
fi

# upsample 8k audio to 16k and generate 16k audio (Because XLSR53 is trained using 16k audio)
if [ ! -z $step03 ]; then
  for dset in "${train_set}" "${valid_set}" ${test_sets};do
	
	mkdir -p "${dump_dir}/${upsample}/${dset}"
	cp ${dump_dir}/${downsample}/${dset}/{spk2utt,text,utt2IntLabel,utt2dur,utt2spk,wav.scp} ${dump_dir}/${upsample}/${dset}/  
	utils/data/resample_data_dir.sh 16000 "${dump_dir}/${upsample}/${dset}"
        mv ${dump_dir}/${upsample}/${dset}/wav.scp  ${dump_dir}/${upsample}/${dset}/wav.scp.tmp 
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${cmd}" \
                    --audio-format "${audio_format}" --fs "${upsample}" \
                    "${dump_dir}/${upsample}/${dset}/wav.scp.tmp" "${dump_dir}/${upsample}/${dset}"
  done	
  for dset in ${test_sets};do
	for snrs in 5 10 15 20;do
            ddset=${dset}_${snrs}_snrs
	    mkdir -p "${dump_dir}/${upsample}/${ddset}"
            cp ${dump_dir}/${downsample}/${ddset}/{spk2utt,text,utt2dur,utt2spk,wav.scp} ${dump_dir}/${upsample}/${ddset}/  
	    utils/data/resample_data_dir.sh 16000 "${dump_dir}/${upsample}/${ddset}"
            mv ${dump_dir}/${upsample}/${ddset}/wav.scp  ${dump_dir}/${upsample}/${ddset}/wav.scp.tmp 
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${cmd}" \
                    --audio-format "${audio_format}" --fs "${upsample}" \
                    "${dump_dir}/${upsample}/${ddset}/wav.scp.tmp" "${dump_dir}/${upsample}/${ddset}"

	done

  done
 
fi


