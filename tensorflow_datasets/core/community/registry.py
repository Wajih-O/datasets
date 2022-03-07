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

"""Meta register that uses other registers."""

import dataclasses
import os
from typing import Any, List, Mapping, Type

from etils import epath
from tensorflow_datasets.core import dataset_builder
# Make sure that github paths are registered
from tensorflow_datasets.core import github_api  # pylint: disable=unused-import
from tensorflow_datasets.core import registered
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.community import register_base
from tensorflow_datasets.core.community import register_package
from tensorflow_datasets.core.community import register_path
import toml


def _as_path_list(path_or_paths: utils.ListOrElem[str]) -> List[epath.Path]:
  if isinstance(path_or_paths, list):
    return [epath.Path(p) for p in path_or_paths]
  else:
    return [epath.Path(path_or_paths)]


@dataclasses.dataclass()
class DatasetRegistry(register_base.BaseRegister):
  """Registry of dataset registries.

  Each namespace is associated with one or more registers. Those registers can
  be any type of register, e.g. code on Github or datasets stored somewhere.

  Attributes:
    registers_per_namespace: per namespace a list of registers it consists of.
  """
  registers_per_namespace: Mapping[str, List[register_base.BaseRegister]]

  def has_namespace(self, namespace: str) -> bool:
    return namespace in self.registers_per_namespace

  def list_namespaces(self) -> List[str]:
    return list(self.registers_per_namespace.keys())

  def list_builders(self) -> List[str]:
    builders = []
    for registers in self.registers_per_namespace.values():
      for register in registers:
        builders.extend(register.list_builders())
    return builders

  def _get_registers(
      self, name: utils.DatasetName) -> List[register_base.BaseRegister]:
    if not self.has_namespace(name.namespace):
      raise registered.DatasetNotFoundError(
          f'Namespace {name.namespace} not found. Should be one of: '
          f'{sorted(self.registers_per_namespace.keys())}')
    return self.registers_per_namespace[name.namespace]

  def builder_cls(
      self,
      name: utils.DatasetName,
  ) -> Type[dataset_builder.DatasetBuilder]:
    """Loads the builder class for the given dataset.

    Arguments:
      name: the name and namespace of the dataset to load the builder class for.

    Returns:
      DatasetNotFoundError if data is not found.
    """
    registers = self._get_registers(name)

    # Typically there's only 1, so add special case so that more informative
    # exceptions are raised.
    if len(registers) == 1:
      return registers[0].builder_cls(name)

    # If this dataset has multiple registers, use the first that can be found.
    for register in registers:
      try:
        return register.builder_cls(name)
      except registered.DatasetNotFoundError:
        pass

    raise registered.DatasetNotFoundError(
        f'Namespace {name.namespace} found, '
        f'but could not load dataset {name.name}.')

  def builder(
      self,
      name: utils.DatasetName,
      **builder_kwargs: Any,
  ) -> dataset_builder.DatasetBuilder:
    """Loads the builder class for the given dataset."""
    registers = self._get_registers(name)

    # Typically there's only 1, so add special case so that more informative
    # exceptions are raised.
    if len(registers) == 1:
      return registers[0].builder(name, **builder_kwargs)

    # If this dataset has multiple registers, use the first that loads.
    for register in registers:
      try:
        return register.builder(name, **builder_kwargs)
      except registered.DatasetNotFoundError:
        pass

    raise registered.DatasetNotFoundError(
        f'Namespace {name.namespace} found with {len(registers)} registers, '
        f'but could not load dataset {name.name}.')

  def get_builder_root_dirs(self, name: utils.DatasetName) -> List[epath.Path]:
    """Returns root dir of the generated builder (without version/config)."""
    result = []
    registers = self.registers_per_namespace[name.namespace]
    for register in registers:
      if isinstance(register, register_path.DataDirRegister):
        result.extend(register.get_builder_root_dirs(name))
      else:
        raise RuntimeError(f'Not supported for non datadir registers ({name})!')
    return result


def _load_register_for_paths(
    namespace: str,
    paths: List[epath.Path],
) -> List[register_base.BaseRegister]:
  """Returns a list of registers for the given paths."""
  code_paths = []
  data_paths = []
  for path in paths:
    if os.fspath(path).startswith('github'):
      code_paths.append(path)
    else:
      data_paths.append(path)

  registers = []
  if data_paths:
    register = register_path.DataDirRegister(
        namespace_to_data_dirs={namespace: data_paths})
    registers.append(register)
  for code_path in code_paths:
    registers.append(register_package.PackageRegister(code_path))
  return registers


def registry_for_config(config_path: epath.Path) -> DatasetRegistry:
  """Returns the registry containing all repositories in the given config.

  The config file should be in `.toml` format and have the follow structure:

  ```toml
  [Namespaces]
  huggingface='github://huggingface/datasets/tree/master/datasets'
  kaggle=['/path/to/datasets/', '/path/to/more_datasets/']
  tensorflow_graphics='gs://tensorflow-graphics/datasets'
  ```

  Arguments:
    config_path: the path that contains the .toml config file.

  Raises:
    RuntimeError: when the config contains errors.
  """
  config = toml.loads(config_path.read_text())
  registers_per_namespace = {}
  for namespace, path_or_paths in config['Namespaces'].items():
    if namespace in registers_per_namespace:
      raise RuntimeError(
          f'Namespace {namespace} is defined twice in config {config_path}')
    registers_per_namespace[namespace] = _load_register_for_paths(
        namespace=namespace, paths=_as_path_list(path_or_paths))
  return DatasetRegistry(registers_per_namespace=registers_per_namespace)


community_register = registry_for_config(
    config_path=(utils.tfds_path() / 'community-datasets.toml'))
