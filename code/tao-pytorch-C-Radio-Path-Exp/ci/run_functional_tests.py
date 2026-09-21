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

"""Run scripts for functional tests."""

import argparse
import os
import subprocess
import sys

from utils import (
    CI, DOCKER_ROOT,
    ROOT_DIR
)

from utils import get_docker_command

TESTS = f"pytest -v --color=yes --testmon {os.path.join(ROOT_DIR, 'tests')}" # -ss


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(
        prog="run_tests_local",
        description="Simple script to run tests.",
        add_help=False)
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag to the local base docker image.",
        type=str)
    parser.add_argument(
        "--skip_slow",
        default=False,
        action="store_true"
    )
    args, unknown_args = parser.parse_known_args(args)
    return vars(args), unknown_args


def main(cl_args=None):
    """Simple function to run local tests."""
    try:
        args, unknown_args = parse_command_line(cl_args)
        manifest_file = os.path.join(ROOT_DIR, "docker/manifest.json")
        docker_command_prefix, docker_image = get_docker_command(
            manifest_file, args["tag"]
        )
        launcher_command = " ".join([TESTS] + unknown_args)
        if args["skip_slow"]:
            print("Skipping slow tests.")
            launcher_command += " -m \"not slow\""
        tao_core_path = "/tao-pt/tao-core"
        if not CI:
            # To build the required libraries.
            launcher_command = f"python setup.py develop && {launcher_command}"
            # We append tao-core to the pythonpath so imports will work
            # The functional tests are run on the base container which does not have the core wheel installed
            launcher_command = "{} -v {}:{} -e PYTHONPATH={}:{} {} bash -c \'{} \'".format(
                docker_command_prefix, ROOT_DIR,
                DOCKER_ROOT, tao_core_path, DOCKER_ROOT,
                docker_image, launcher_command)
        print(launcher_command)
        subprocess.check_call(launcher_command, stdout=sys.stdout, stderr=sys.stdout, shell=True)
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        raise RuntimeError(f"Tests failed execution") from e
    except Exception as exc:
        raise RuntimeError(f"Tests failed with error {exc}") from exc


if __name__=="__main__":
    main(sys.argv[1:])
