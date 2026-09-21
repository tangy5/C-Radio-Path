# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import encrypt_checkpoint
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint


@pytest.mark.cv_unit
def test_patch_decrypt_ckpt():
    fake_ckpt = {"state_dict": {"a": 1, "b": 2, "c": 3}}

    encrypted_ckpt = encrypt_checkpoint(fake_ckpt, "tao")

    assert encrypted_ckpt["state_dict_encrypted"] is not False, f"encrypted_ckpt[\"state_dict_encrypted\"] "\
        f"should be True. Got False."

    decrypted_ckpt = patch_decrypt_checkpoint(encrypted_ckpt, "tao")

    assert decrypted_ckpt["state_dict_encrypted"] is False, f"encrypted_ckpt[\"state_dict_encrypted\"] "\
        f"should be False. Got True."
