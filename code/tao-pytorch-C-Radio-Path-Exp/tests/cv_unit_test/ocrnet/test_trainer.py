# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import lmdb
import os
from PIL import Image
import pytest
import tempfile
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from nvidia_tao_core.config.ocrnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.pl_ocr_data_module import OCRDataModule
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel

DEFAULT_LABEL= "0123456789abcdefghijklmnopqrstuvwxyz"
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 200
FAST_DEV_RUN = 2  # Run dry run 2 times
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
lmdb_dir = os.path.join(tmp_top_dir, "lmdb")
gt_file = os.path.join(tmp_top_dir, "gt.txt")
character_file = os.path.join(tmp_top_dir, "character_list")
batch_size = 32


@pytest.fixture
def _test_data():
    os.makedirs(tmp_top_dir, exist_ok=True)
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8))
    sample_cnt = batch_size
    tmp_img_path = os.path.join(tmp_top_dir, "tmp_img.png")
    img.save(tmp_img_path)
    with open(tmp_img_path, 'rb') as f:
        img_bin = f.read()

    cache = {}
    for i in range(1, sample_cnt + 1):
        imageKey = 'image-%09d'.encode() % i
        labelKey = 'label-%09d'.encode() % i
        cache[imageKey] = img_bin
        cache[labelKey] = DEFAULT_LABEL.encode()

    cache['num-samples'.encode()] = str(sample_cnt).encode()
    os.makedirs(lmdb_dir, exist_ok=True)
    env = lmdb.open(lmdb_dir, map_size=1099511627776)

    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

    os.makedirs(os.path.join(tmp_top_dir, 'images'), exist_ok=True)
    with open(gt_file, "w") as f:
        for idx in range(sample_cnt):
            tmp_img_path = os.path.join(tmp_top_dir, 'images', f"tmp_img_{idx}.png")
            f.write(f"{tmp_img_path} {DEFAULT_LABEL}\n")
            img.save(tmp_img_path)

    with open(character_file, "w") as f:
        for ch in DEFAULT_LABEL:
            f.write(f"{ch}\n")


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1

    experiment_config.dataset.character_list_file = character_file
    experiment_config.dataset.max_label_length = len(DEFAULT_LABEL)

    experiment_config.model.TPS = True
    experiment_config.model.input_width = DEFAULT_WIDTH
    experiment_config.model.input_height = DEFAULT_HEIGHT

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1

    experiment_config.dataset.character_list_file = character_file
    experiment_config.dataset.max_label_length = len(DEFAULT_LABEL)

    experiment_config.model.TPS = True
    experiment_config.model.input_width = DEFAULT_WIDTH
    experiment_config.model.input_height = DEFAULT_HEIGHT

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.inference_dataset_dir = tmp_top_dir

    experiment_config.dataset.character_list_file = character_file

    experiment_config.model.TPS = True
    experiment_config.model.input_width = DEFAULT_WIDTH
    experiment_config.model.input_height = DEFAULT_HEIGHT

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.ocrnet
@pytest.mark.train
@pytest.mark.parametrize("backbone", ["FAN_tiny_2X", "ResNet"])
@pytest.mark.parametrize("dataset", ["lmdb", "raw"])
def test_trainer_fit(_test_data, _train_spec, backbone, dataset):

    _train_spec.model.backbone = backbone
    if backbone == 'FAN_tiny_2X':
        _train_spec.model.prediction = "Attn"
    else:
        _train_spec.model.prediction = "CTC"

    if dataset == 'lmdb':
        _train_spec.dataset.train_dataset_dir = [lmdb_dir]
        _train_spec.dataset.val_dataset_dir = lmdb_dir
    else:
        _train_spec.dataset.train_dataset_dir = [tmp_top_dir]
        _train_spec.dataset.train_gt_file = gt_file
        _train_spec.dataset.val_dataset_dir = tmp_top_dir
        _train_spec.dataset.val_gt_file = gt_file

    dm = OCRDataModule(_train_spec)
    dm.setup(stage='fit')
    model = OCRNetModel(_train_spec, dm)
    clip_grad = _train_spec.train.clip_grad_norm

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      gradient_clip_val=clip_grad,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.ocrnet
@pytest.mark.evaluate
@pytest.mark.parametrize("backbone", ["FAN_tiny_2X", "ResNet"])
@pytest.mark.parametrize("dataset", ["lmdb", "raw"])
def test_trainer_evaluate(_test_data, _eval_spec, backbone, dataset):

    _eval_spec.model.backbone = backbone
    if backbone == 'FAN_tiny_2X':
        _eval_spec.model.prediction = "Attn"
    else:
        _eval_spec.model.prediction = "CTC"

    if dataset == 'lmdb':
        _eval_spec.evaluate.test_dataset_dir = lmdb_dir
    else:
        _eval_spec.evaluate.test_dataset_dir = tmp_top_dir
        _eval_spec.evaluate.test_dataset_gt_file = gt_file

    dm = OCRDataModule(_eval_spec)
    dm.setup(stage='test')
    model = OCRNetModel(_eval_spec, dm)

    trainer = Trainer(devices=_eval_spec.train.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN)

    # Test evaluate
    trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.ocrnet
@pytest.mark.inference
@pytest.mark.parametrize("backbone", ["FAN_tiny_2X", "ResNet"])
def test_trainer_inference(_test_data, _infer_spec, backbone):

    _infer_spec.model.backbone = backbone
    if backbone == 'FAN_tiny_2X':
        _infer_spec.model.prediction = "Attn"
    else:
        _infer_spec.model.prediction = "CTC"

    dm = OCRDataModule(_infer_spec)
    dm.setup(stage='predict')
    model = OCRNetModel(_infer_spec, dm)

    trainer = Trainer(devices=_infer_spec.train.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN)

    # Test predict
    trainer.predict(model, dm)

    tmp_top_obj.cleanup()
