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
"""A module to record audio in a streaming basis."""
import threading
import sounddevice as sd

from tensorflow_lite_support.python.task.audio.core import tensor_audio


class AudioRecord(object):
  """A class to record audio in a streaming basis."""

  def __init__(self, tensor: tensor_audio.TensorAudio) -> None:
    self._lock = threading.Lock()
    self._tensor = tensor
    self._audio_buffer = tensor.get_buffer()

    input_sample_count = self._tensor.get_sample_count()
    input_audio_format = self._tensor.get_format()

    def audio_callback(audio_data, *_):
      """A callback to receive recorded audio data from sounddevice."""
      self._lock.acquire()
      if len(audio_data) > input_sample_count:
        # Only take the latest input if the audio data received is
        # longer than what the TensorAudio can store.
        self._tensor.load_from_array(audio_data[-input_sample_count:])
      else:
        self._tensor.load_from_array(audio_data)
      self._lock.release()

    # Create an input stream to continuously capture the audio data.
    self._stream = sd.InputStream(
      channels=input_audio_format.channels,
      samplerate=input_audio_format.sample_rate,
      callback=audio_callback,
    )

  def get_tensor_audio(self) -> tensor_audio.TensorAudio:
    """Gets the TensorAudio object."""
    return self._tensor

  def start_recording(self) -> None:
    """Start the audio recording."""
    self._stream.start()

  def stop(self) -> None:
    """Stop the audio recording."""
    self._audio_buffer = []
    self._stream.stop()
