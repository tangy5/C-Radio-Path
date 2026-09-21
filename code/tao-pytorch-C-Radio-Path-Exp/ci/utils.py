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

"""Utilities to run tests for TLT."""

import json
import os
import platform


def configure_env():
    """Get the env configuration."""
    ci = False
    if os.getenv("CI_PROJECT_DIR", None) is not None:
        root_dir = os.getenv("CI_PROJECT_DIR", None)
        docker_root = root_dir
        ci = True
    else:
        # Directory for local test runs.
        root_dir = os.getenv(
            "NV_TAO_PYTORCH_TOP",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        docker_root = "/tao-pt"
    return ci, docker_root, root_dir


CI, DOCKER_ROOT, ROOT_DIR = configure_env()
# Testing modules for the NLP, ASR, SUBTASKS and core.
RCFILE = ".pylintrc"
TEST_MODULES = [
    "nvidia_tao_pytorch/cv",
    "nvidia_tao_pytorch/core",
    "nvidia_tao_pytorch/pointcloud",
    "nvidia_tao_pytorch/pruning",
    "nvidia_tao_pytorch/sdg",
    "nvidia_tao_pytorch/ssl"
]


def get_docker_information(manifest_file):
    """Loading the manifest to pick up the latest base docker."""
    assert os.path.exists(manifest_file), (
        "Manifest file doesn't exist at {}".format(
            manifest_file)
    )
    with open(manifest_file, "r") as m_file:
        docker_config = json.load(m_file)

    # Platform keys for digest lookup
    X86_KEY = "x86"
    ARM_KEY = "arm"

    # Handle both old and new manifest formats
    if "digest" in docker_config:
        # Old format with single digest
        digest = docker_config["digest"]
    elif "digests" in docker_config:
        # New format with platform-specific digests
        arch = platform.machine()
        if arch == "x86_64":
            digest = docker_config["digests"][X86_KEY]
        elif arch == "aarch64":
            digest = docker_config["digests"][ARM_KEY]
        else:
            # Fallback to x86
            digest = docker_config["digests"][X86_KEY]
    else:
        raise ValueError("Invalid manifest format: missing 'digest' or 'digests' field")

    return docker_config["registry"], docker_config["repository"], digest


def get_docker_command(manifest_file, tag):
    """Get formatted docker command."""
    docker_registry, docker_repository, docker_digest = get_docker_information(
        manifest_file)
    docker_image = "{}/{}@{}".format(
        docker_registry, docker_repository, docker_digest
    )
    if tag is not None:
        docker_image = "{}/{}:{}".format(
            docker_registry, docker_repository, tag
        )
    docker_command_prefix = "docker run --rm --gpus all"
    return docker_command_prefix, docker_image
