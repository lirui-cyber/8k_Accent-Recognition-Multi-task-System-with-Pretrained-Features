import argparse
import numpy as np
import s3prl.upstream.wavlm.hubconf as wavlmhubconf
import s3prl.upstream.wav2vec2.hubconf as wav2vec2hubconf
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import json
from kaldiio import  WriteHelper
import logging
import os
import librosa
import sys
logging.basicConfig(format='%(asctime)s %(message)s')

class RawFeatures(data.Dataset):
    def __init__(self,uttid, wav_path):
        self.uttid_list = uttid
        self.wavpath_list = wav_path
    def __getitem__(self, index):
        wav, sr = librosa.load(self.wavpath_list[index], sr=8000)
        id = self.uttid_list[index]
        return id, wav
    def __len__(self):
        return len(self.uttid_list)

def prepare_data(wav_scp):
    with open(wav_scp, 'r') as f:
        lines_wav = f.readlines()
        audio_list = [x.split()[-1].strip() for x in lines_wav]
        name_list = [x.split()[0].strip() for x in lines_wav]
    train_set = RawFeatures(name_list, audio_list)
    trainloader = DataLoader(dataset=train_set,
                             batch_size=1,
                             pin_memory=True,
                             num_workers=1)
    return trainloader
def feat_extract(dataloader, model, device, feat_layer,save_dir):
    feat_scp_path = "{}.scp".format(os.path.join(save_dir, "feats"))
    feat_ark_path = "{}.ark".format(os.path.join(save_dir, "feats"))
    total = 0
    model.eval()
    with WriteHelper('ark,scp:' + feat_ark_path + "," + feat_scp_path) as writer:
        with torch.no_grad():
            for step, (uttid, wav) in enumerate(dataloader):
                wav = torch.tensor(wav).to(device=device, dtype=torch.float)
                features = model(wav)["hidden_state_{}".format(feat_layer)]
                features_ = features.squeeze(0).cpu().detach().numpy()
                iid = uttid[0]
                writer(iid,features_)
                print(iid +" "+"feature extracted")
                total += 1
    print("*************** {} finished! Total extracted features :{}".format(save_dir,total))

def main():
    device = torch.device("cuda")
    dump = sys.argv[1]
    feat_layer = sys.argv[2]  
    if "wavlm" in dump:
        model_path = "pretrained-model/wavlm_large.pt"
        model = wavlmhubconf.wavlm_local(ckpt=model_path)
    else:
        model_path = "pretrained-model/xlsr_53_56k.pt" 
        model =wav2vec2hubconf.wav2vec2_local(ckpt=model_path)
    model.to(device)
    data_set="train cv_all test test_20_snrs test_15_snrs test_10_snrs test_5_snrs"     
    test_sets = data_set.split()
    for test in test_sets:
        save_dir = dump + "/" + test
        wav_scp = dump + "/" + test + "/wav.scp"
        testloader = prepare_data(wav_scp)
        feat_extract(testloader,model,device,feat_layer,save_dir)

if __name__ == '__main__':
    main()
