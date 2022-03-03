# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""TensorAudio class."""
from typing import NamedTuple

import numpy as np

from tensorflow_lite_support.python.task.audio.core.pybinds import audio_utils


class TensorAudio(object):
  """A wrapper class to store the input audio."""

  def __init__(self,
               audio_data,
               is_from_file: bool = False) -> None:
    """Initializes the `TensorAudio` object.

    Args:
      audio_data: AudioBuffer, contains raw audio data, audio format
       and buffer size info.
      is_from_file: boolean, whether `image_data` is loaded from the image file,
        if True, need to free the storage of AudioBuffer in the destructor.
    """
    self._audio_data = audio_data
    self._is_from_file = is_from_file

    # Gets the AudioBuffer object.

  def __del__(self):
    """Clear the internal buffer and fill it with zeros."""
    self._audio_data.fill(0)

  @classmethod
  def from_file(cls, file_name: str) -> "TensorAudio":
    """Creates `TensorAudio` object from the audio file.

    Args:
      file_name: Image file name.

    Returns:
      `TensorAudio` object.

    Raises:
      status.StatusNotOk if the image file can't be decoded. Need to import
        the module to catch this error: `from pybind11_abseil import status`,
        see https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    audio_data = audio_utils.DecodeAudioFromWaveFile(file_name)
    return cls(audio_data, is_from_file=True)

  def get_buffer(self) -> np.ndarray:
    """Gets the numpy array that represents `self._audio_data`.
    Returns:
      Numpy array that represents `self._audio_data` which is an
        `audio_buffer.AudioBuffer` object. To avoid copy, we will use
        `return np.array(..., copy = False)`. Therefore, this `TensorAudio`
        object should outlive the returned numpy array.
    """
    return np.array(self._audio_data, copy=False)
