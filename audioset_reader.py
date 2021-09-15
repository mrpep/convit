from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import glob
from pathlib import Path

import torchaudio

class AudiosetDataset(Dataset):
    def __init__(self, wav_dir, metadata_dir,melbins=64, target_length=992, specaugment = dict(f=24,t=192), norm_stats = dict(mean=-4.2677393,std=4.5689974)):
        self.melbins = melbins
        self.target_length = target_length
        csv_kwargs = dict(delimiter = ',', skipinitialspace = True, quotechar = '"', encoding = 'utf-8', header = 2)
        CLASS_LABEL_MAP = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        self.df_metadata = pd.read_csv(metadata_dir, **csv_kwargs)
        self.df_metadata = self.df_metadata.set_index('# YTID')
        self.class_map = pd.read_csv(CLASS_LABEL_MAP)
        self.class_map = self.class_map.set_index('mid')
        self.specaugment = specaugment
        self.norm_stats = norm_stats

        print('Processing labels')
        self.make_labels()
        available_files = list(glob.glob(wav_dir + '/*.wav'))
        available_stems = [Path(x).stem for x in available_files]

        original_len = len(self.df_metadata)
        self.df_metadata = self.df_metadata.loc[available_stems]
        self.df_metadata['filename'] = available_files
        print('{}/{} audio files are available'.format(len(self.df_metadata),original_len))

    def make_labels(self):
        def fn(x):
            y = np.zeros((len(self.class_map)))
            for l in x.split(','):
                y[int(self.class_map.loc[l]['index'])] = 1
            return y
        self.df_metadata['target'] = self.df_metadata['positive_labels'].apply(fn)

    def __len__(self):
        return len(self.df_metadata)

    def __getitem__(self, idx):
        row = self.df_metadata.iloc[idx]

        #Read and extract features
        waveform, sr = torchaudio.load(row['filename'])
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
        
        #Normalize
        fbank = (fbank - self.norm_stats['mean']) / (self.norm_stats['std'] * 2)

        #Specaugment:
        freqm = torchaudio.transforms.FrequencyMasking(self.specaugment['f'])
        timem = torchaudio.transforms.TimeMasking(self.specaugment['t'])
        fbank = torch.transpose(fbank, 0, 1)
        if self.specaugment['f'] != 0:
            fbank = freqm(fbank)
        if self.specaugment['t'] != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = torch.unsqueeze(fbank,0)

        return fbank, row['target']