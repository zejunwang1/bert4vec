# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert RoFormer checkpoint."""


import argparse

import torch
from torch import nn
from transformers import RoFormerConfig, RoFormerForMaskedLM, RoFormerModel, load_tf_weights_in_roformer
from transformers.models.roformer.modeling_roformer import RoFormerOnlyMLMHead
from transformers.utils import logging


logging.set_verbosity_info()

class RoFormerModelWithPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roformer = RoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(config)     
        self.roformer.pooler = nn.Linear(config.hidden_size, config.hidden_size)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = RoFormerConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = RoFormerModelWithPooler(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_roformer(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)

# python convert_roformer_sim_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path "./chinese_roformer-sim-char-ft_L-6_H-384_A-6/bert_model.ckpt" --bert_config_file "./chinese_roformer-sim-char-ft_L-6_H-384_A-6/bert_config.json" --pytorch_dump_path "./chinese_roformer-sim-char-ft_L-6_H-384_A-6/pytorch_model.bin"
# python convert_roformer_sim_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path "./chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt" --bert_config_file "./chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json" --pytorch_dump_path "./chinese_roformer-sim-char-ft_L-12_H-768_A-12/pytorch_model.bin"