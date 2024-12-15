# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: For EfficientNet V2 utils.
# Create: 2021-12-2

"""
Code based on https://github.com/lingffff/EfficientNetV2-PyTorch/blob/main/utils.py and slightly modified
"""

import re
import collections

################################################################################
# Helper functions for loading model params
################################################################################

# BlockDecoder: A Class for encoding and decoding BlockArgs
# efficientnet_params: A function to query compound coefficient
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'dropout_rate', 'num_classes'
])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'fused'
])

class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            fused=('f' in block_string)
        )

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """

        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def get_efficientnetv2_params(model_name, num_classes):
    #################### EfficientNet V2 configs ####################
    v2_base_block = [  # The baseline config for v2 models.
        'r1_k3_s1_e1_i32_o16_f',
        'r2_k3_s2_e4_i16_o32_f',
        'r2_k3_s2_e4_i32_o48_f',
        'r3_k3_s2_e4_i48_o96_se0.25',
        'r5_k3_s1_e6_i96_o112_se0.25',
        'r8_k3_s2_e6_i112_o192_se0.25',
    ]
    v2_s_block = [  # about base * (width1.4, depth1.8)
        'r2_k3_s1_e1_i16_o16_f',
        'r4_k3_s2_e4_i16_o32_f',
        'r4_k3_s2_e4_i32_o48_f',
        'r6_k3_s2_e4_i48_o96_se0.25',
        'r6_k3_s1_e6_i96_o128_se0.25',
        'r10_k3_s2_e6_i128_o160_se0.25',
    ] #ADAPTED FOR CIFAR10
    v2_m_block = [  # about base * (width1.6, depth2.2)
        'r3_k3_s1_e1_i16_o16_f',
        'r5_k3_s2_e4_i16_o32_f',
        'r5_k3_s2_e4_i32_o64_f',
        'r6_k3_s2_e4_i64_o128_se0.25',
        'r8_k3_s1_e6_i128_o160_se0.25',
        'r12_k3_s2_e6_i160_o240_se0.25',
        'r5_k3_s1_e6_i240_o320_se0.25',
    ] #ADAPTED FOR CIFAR10
    v2_l_block = [  # about base * (width2.0, depth3.1)
        'r4_k3_s1_e1_i32_o32_f',
        'r7_k3_s2_e4_i32_o64_f',
        'r7_k3_s2_e4_i64_o96_f',
        'r10_k3_s2_e4_i96_o192_se0.25',
        'r19_k3_s1_e6_i192_o224_se0.25',
        'r25_k3_s2_e6_i224_o384_se0.25',
        'r7_k3_s1_e6_i384_o640_se0.25',
    ]

    efficientnetv2_params = {
        # (block, width, depth, dropout)
        'efficientnetv2-s': (v2_s_block, 1.0, 1.0, 0.2),  # Small
        'efficientnetv2-m': (v2_m_block, 1.0, 1.0, 0.3),  # Medium
        'efficientnetv2-l': (v2_l_block, 1.0, 1.0, 0.4),  # Large
    }

    assert model_name in list(efficientnetv2_params.keys()), "Wrong model name."
    all_params = efficientnetv2_params[model_name]

    blocks_args = BlockDecoder.decode(all_params[0])

    global_params = GlobalParams(
        width_coefficient=all_params[1],
        depth_coefficient=all_params[2],
        dropout_rate=all_params[3],
        num_classes=num_classes,
    )

    return blocks_args, global_params