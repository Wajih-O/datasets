# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
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

"""Tests for registry."""
from unittest import mock

from tensorflow_datasets.core.community import register_package
from tensorflow_datasets.core.community import register_path
from tensorflow_datasets.core.community import registry as registry_lib


def test_load_register_for_path_github():
  registers = registry_lib._load_register_for_paths(
      namespace='huggingface',
      paths=['github://huggingface/datasets/tree/master/datasets'])
  assert len(registers) == 1
  assert isinstance(registers[0], register_package.PackageRegister)


def test_load_register_for_path_gcs():
  registers = registry_lib._load_register_for_paths(
      namespace='my_namespace',
      paths=['gs://my-bucket/datasets', 'gs://my-bucket2/datasets'])
  assert len(registers) == 1
  assert isinstance(registers[0], register_path.DataDirRegister)


def test_load_register_for_path_mixed():
  registers = registry_lib._load_register_for_paths(
      namespace='my_namespace',
      paths=[
          'github://huggingface/datasets/tree/master/datasets',
          'gs://my-bucket/datasets',
      ])
  assert len(registers) == 2
  assert isinstance(registers[0], register_path.DataDirRegister)
  assert isinstance(registers[1], register_package.PackageRegister)


def test_community_register():
  assert 'huggingface' in registry_lib.community_register.list_namespaces()


def test_dataset_registry_list_builders():
  register1 = mock.create_autospec(register_path.DataDirRegister)
  register1.list_builders.return_value = ['a', 'b']
  register2 = mock.create_autospec(register_package.PackageRegister)
  register2.list_builders.return_value = ['c']
  registry = registry_lib.DatasetRegistry(registers_per_namespace={
      'ns1': [register1],
      'ns2': [register2],
  })
  assert set(registry.list_namespaces()) == {'ns1', 'ns2'}
  assert set(registry.list_builders()) == {'a', 'b', 'c'}
