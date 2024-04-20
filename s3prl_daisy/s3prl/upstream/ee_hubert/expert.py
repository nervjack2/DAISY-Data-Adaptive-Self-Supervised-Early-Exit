# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""

import logging
from pathlib import Path

import torch
import numpy as np 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .convert import load_converted_model

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, f_type, ckpt, model_config, **kwargs):
        super().__init__(**kwargs)
        model, task_cfg = load_converted_model(ckpt)
        self.model = model
        self.task_cfg = task_cfg
        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0
        self.f_type = f_type 
        self.model_config = model_config
        # Hyperparameters 
        with open(self.model_config, 'r') as file:
            import yaml
            self.hyper_param = yaml.load(file, Loader=yaml.FullLoader)

        self.ext_way = self.hyper_param['ext_way']
        self.u_threshold = self.hyper_param['u_threshold'] 
        self.inf_threshold = self.hyper_param.get('inf_threshold', 0.2)
        # These four hyperparameters would be determined during training. 
        # The program will automatically save these four parameters into model_config.  
        # They are needed for inference.
        self.n_sample = self.hyper_param.get('n_sample', 0)
        self.n_layer = self.hyper_param.get('n_layer', [0 for i in range(12)])
        self.min_layer = self.hyper_param.get('min_layer', 12)
        self.max_layer = self.hyper_param.get('max_layer', 0)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, mode='train'):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        
        if mode == 'train':
            ext_range = None
        else:
            if self.ext_way == 'avg':
                total_n_layer = sum([self.n_layer[i]*(i+1) for i in range(12)])
                avg_layer = total_n_layer / max(self.n_sample,1)
                ext_range = {}
                for i in range(int(np.floor(avg_layer)), int(np.ceil(avg_layer))+1):
                    ext_range[i] = 1
            elif self.ext_way == 'min-max':
                ext_range = {}
                for i in range(self.min_layer, self.max_layer+1):
                    ext_range[i] = 1
            elif self.ext_way == 'threshold': 
                prob_layer = [self.n_layer[i]/max(self.n_sample,1) for i in range(12)]
                ext_range = {}
                for i, prob in enumerate(prob_layer):
                    if prob > self.inf_threshold:
                        ext_range[i+1] = 1
            else:
                print(f"Type error {self.ext_way}")
                exit(0)

        layer_results, x, conv_feat, padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
            f_type=self.f_type,
            mode=mode,
            ext_range=ext_range,
            u_threshold=self.u_threshold
        )
        
        n_layer = len(layer_results)

        if mode == 'train':
            if n_layer < self.min_layer:
                self.min_layer = n_layer
            if n_layer > self.max_layer:
                self.max_layer = n_layer
            self.n_layer[n_layer-1] += 1
            self.n_sample += 1

        w_features = [conv_feat] + layer_results

        states = {
            "hidden_states": w_features
        }

        return states



class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, f_type, ckpt, **kwargs):
        super().__init__(**kwargs)
        logger.warning("Use the legacy expert for HuBERT which depends on fairseq")
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.task = task

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks


class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        logger.warning("Use the legacy expert for HuBERT which depends on fairseq")
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.model.feature_grad_mult = 0.0
        self.task = task

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

        self._init_layerdrop = self.model.encoder.layerdrop

    @property
    def layer_drop(self):
        return self.model.encoder.layerdrop

    def set_layer_drop(self, layerdrop: float = None):
        if isinstance(layerdrop, float):
            self.model.encoder.layerdrop = layerdrop
        elif layerdrop is None:
            self.model.encoder.layerdrop = self._init_layerdrop
        else:
            raise ValueError("layerdrop can only be float or None")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
