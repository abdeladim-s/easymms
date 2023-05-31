#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Constants
"""
import logging
from pathlib import Path

import numpy as np
from platformdirs import *

PACKAGE_NAME = 'easymms'
LOGGING_LEVEL = logging.INFO

PACKAGE_DATA_DIR = user_data_dir(PACKAGE_NAME)


TTS_DIR = (Path(PACKAGE_DATA_DIR) / 'tts').resolve()
TTS_MODELS_BASE_URL = "https://dl.fbaipublicfiles.com/mms/tts/"  # lang.tar.gz
VITS_URL = "https://github.com/jaywalnut310/vits"
VITS_DIR = TTS_DIR / 'vits'

FAIRSEQ_URL = "https://github.com/facebookresearch/fairseq"
FAIRSEQ_DIR = Path(PACKAGE_DATA_DIR) / 'fairseq'

CFG = {
  '_name': None,
  'task': {
    '_name': 'audio_finetuning',
    'data': '',
    'labels': 'ltr'
  },
  'decoding': {
    '_name': None,
    'nbest': 1,
    'unitlm': False,
    'lmpath': '???',
    'lexicon': None,
    'beam': 50,
    'beamthreshold': 50.0,
    'beamsizetoken': None,
    'wordscore': -1.0,
    'unkweight': -np.inf,
    'silweight': 0.0,
    'lmweight': 2.0,
    'type': 'viterbi',
    'unique_wer_file': False,
    'results_path': ''
  },
  'common': {
    '_name': None,
    'no_progress_bar': False,
    'log_interval': 100,
    'log_format': None,
    'log_file': None,
    'aim_repo': None,
    'aim_run_hash': None,
    'tensorboard_logdir': None,
    'wandb_project': None,
    'azureml_logging': False,
    'seed': 1,
    'cpu': False,
    'tpu': False,
    'bf16': False,
    'memory_efficient_bf16': False,
    'fp16': False,
    'memory_efficient_fp16': False,
    'fp16_no_flatten_grads': False,
    'fp16_init_scale': 128,
    'fp16_scale_window': None,
    'fp16_scale_tolerance': 0.0,
    'on_cpu_convert_precision': False,
    'min_loss_scale': 0.0001,
    'threshold_loss_scale': None,
    'amp': False,
    'amp_batch_retries': 2,
    'amp_init_scale': 128,
    'amp_scale_window': None,
    'user_dir': None,
    'empty_cache_freq': 0,
    'all_gather_list_size': 16384,
    'model_parallel_size': 1,
    'quantization_config_path': None,
    'profile': False,
    'reset_logging': False,
    'suppress_crashes': False,
    'use_plasma_view': False,
    'plasma_path': '/tmp/plasma'
  },
  'common_eval': {
    '_name': None,
    'path': '',
    'post_process': 'letter',
    'quiet': False,
    'model_overrides': '{}',
    'results_path': None
  },
  'checkpoint': {
    '_name': None,
    'save_dir': 'checkpoints',
    'restore_file': 'checkpoint_last.pt',
    'continue_once': None,
    'finetune_from_model': None,
    'reset_dataloader': False,
    'reset_lr_scheduler': False,
    'reset_meters': False,
    'reset_optimizer': False,
    'optimizer_overrides': '{}',
    'save_interval': 1,
    'save_interval_updates': 0,
    'keep_interval_updates': -1,
    'keep_interval_updates_pattern': -1,
    'keep_last_epochs': -1,
    'keep_best_checkpoints': -1,
    'no_save': False,
    'no_epoch_checkpoints': False,
    'no_last_checkpoints': False,
    'no_save_optimizer_state': False,
    'best_checkpoint_metric': 'loss',
    'maximize_best_checkpoint_metric': False,
    'patience': -1,
    'checkpoint_suffix': '',
    'checkpoint_shard_count': 1,
    'load_checkpoint_on_all_dp_ranks': False,
    'write_checkpoints_asynchronously': False,
    'model_parallel_size': 1
  },
  'distributed_training': {
    '_name': None,
    'distributed_world_size': 1,
    'distributed_num_procs': 1,
    'distributed_rank': 0,
    'distributed_backend': 'nccl',
    'distributed_init_method': None,
    'distributed_port': -1,
    'device_id': 0,
    'distributed_no_spawn': False,
    'ddp_backend': 'legacy_ddp',
    'ddp_comm_hook': 'none',
    'bucket_cap_mb': 25,
    'fix_batches_to_gpus': False,
    'find_unused_parameters': False,
    'gradient_as_bucket_view': False,
    'fast_stat_sync': False,
    'heartbeat_timeout': -1,
    'broadcast_buffers': False,
    'slowmo_momentum': None,
    'slowmo_base_algorithm': 'localsgd',
    'localsgd_frequency': 3,
    'nprocs_per_node': 1,
    'pipeline_model_parallel': False,
    'pipeline_balance': None,
    'pipeline_devices': None,
    'pipeline_chunks': 0,
    'pipeline_encoder_balance': None,
    'pipeline_encoder_devices': None,
    'pipeline_decoder_balance': None,
    'pipeline_decoder_devices': None,
    'pipeline_checkpoint': 'never',
    'zero_sharding': 'none',
    'fp16': False,
    'memory_efficient_fp16': True,
    'tpu': False,
    'no_reshard_after_forward': False,
    'fp32_reduce_scatter': False,
    'cpu_offload': False,
    'use_sharded_state': False,
    'not_fsdp_flatten_parameters': False
  },
  'dataset': {
    '_name': None,
    'num_workers': 1,
    'skip_invalid_size_inputs_valid_test': False,
    'max_tokens': 4000000,
    'batch_size': None,
    'required_batch_size_multiple': 1,
    'required_seq_len_multiple': 1,
    'dataset_impl': None,
    'data_buffer_size': 10,
    'train_subset': 'train',
    'valid_subset': 'valid',
    'combine_valid_subsets': None,
    'ignore_unused_valid_subsets': False,
    'validate_interval': 1,
    'validate_interval_updates': 0,
    'validate_after_updates': 0,
    'fixed_validation_seed': None,
    'disable_validation': False,
    'max_tokens_valid': 4000000,
    'batch_size_valid': None,
    'max_valid_steps': None,
    'curriculum': 0,
    'gen_subset': 'eng:dev',
    'num_shards': 1,
    'shard_id': 0,
    'grouped_shuffling': False,
    'update_epoch_batch_itr': False,
    'update_ordered_indices_seed': False
  },
  'is_ax': False
}

HYPO_WORDS_FILE = 'hypo.word'

ALIGNMENT_MODEL_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
ALIGNMENT_DICTIONARY_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt"

UROMAN_URL = "https://github.com/isi-nlp/uroman"
UROMAN_DIR = Path(PACKAGE_DATA_DIR) / 'uroman'

MMS_LANGS_FILE = (Path(__file__).parent / 'data' / 'mms_langs.json').resolve()
