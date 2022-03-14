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
"""Audio classifier task."""

import dataclasses
import threading

import sounddevice as sd
from typing import Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.core import task_utils
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.processor.proto import embeddings_pb2
from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core.pybinds import _pywrap_audio_buffer
from tensorflow_lite_support.python.task.audio.pybinds import _pywrap_audio_embedder
from tensorflow_lite_support.python.task.audio.pybinds import audio_embedder_options_pb2

_CppAudioFormat = _pywrap_audio_buffer.AudioFormat
_ProtoAudioEmbedderOptions = audio_embedder_options_pb2.AudioEmbedderOptions
_CppAAudioEmbedder = _pywrap_audio_embedder.AudioEmbedder


@dataclasses.dataclass
class AudioEmbedderOptions:
  """Options for the audio embedder task."""
  base_options: task_options.BaseOptions
  embedding_options: Optional[embedding_options_pb2.EmbeddingOptions] = None


class AudioEmbedder(object):
  """Class that performs dense feature vector extraction on audio."""

  def __init__(self, options: AudioEmbedderOptions,
               cpp_embedder: _CppAAudioEmbedder) -> None:
    """Initializes the `AudioEmbedder` object."""
    self._options = options
    self._embedder = cpp_embedder

  @classmethod
  def create_from_options(cls, options: AudioEmbedderOptions) -> "AudioEmbedder":
    """Creates the `AudioEmbedder` object from audio embedder options.
    Args:
      options: Options for the audio embedder task.
    Returns:
      `AudioEmbedder` object that's created from `options`.
    Raises:
      TODO(b/220931229): Raise RuntimeError instead of status.StatusNotOk.
      status.StatusNotOk if failed to create `AudioEmbedder` object from
        `AudioEmbedderOptions` such as missing the model. Need to import the
        module to catch this error: `from pybind11_abseil import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    # Creates the object of C++ AudioEmbedder class.
    proto_options = _ProtoAudioEmbedderOptions()
    proto_options.base_options.CopyFrom(
      task_utils.ConvertToProtoBaseOptions(options.base_options))
    if options.embedding_options:
      embedding_options = proto_options.embedding_options.add()
      embedding_options.CopyFrom(options.embedding_options)
    embedder = _CppAAudioEmbedder.create_from_options(proto_options)
    return cls(options, embedder)

  def create_input_tensor_audio(self) -> tensor_audio.TensorAudio:
    """Creates a TensorAudio instance to store the audio input.
    Returns:
        A TensorAudio instance.
    """
    return tensor_audio.TensorAudio(
      audio_format=self.required_audio_format,
      sample_count=self.required_input_buffer_size)

  def create_input_audio_recorder(
      self
  ) -> (tensor_audio.TensorAudio, sd.InputStream):
    tensor = self.create_input_tensor_audio()

    input_sample_count = tensor.get_sample_count()
    input_audio_format = tensor.get_format()

    tensor = tensor_audio.TensorAudio(
      audio_format=input_audio_format, sample_count=input_sample_count)
    lock = threading.Lock()

    def audio_callback(audio_data, *_):
      """A callback to receive recorded audio data from sounddevice."""
      lock.acquire()
      if len(audio_data) > input_sample_count:
        # Only take the latest input if the audio data received is
        # longer than what the TensorAudio can store.
        tensor.load_from_array(audio_data[-input_sample_count:])
      else:
        tensor.load_from_array(audio_data)
      lock.release()

    # Create an input stream to continuously capture the audio data.
    input_stream = sd.InputStream(
      channels=input_audio_format.channels,
      samplerate=input_audio_format.sample_rate,
      callback=audio_callback,
    )

    return tensor, input_stream

  def embed(
      self,
      audio: tensor_audio.TensorAudio
  ) -> embeddings_pb2.EmbeddingResult:
    """Performs actual feature vector extraction on the provided audio.
    Args:
      audio: Tensor audio, used to extract the feature vectors.
    Returns:
      embedding result.
    Raises:
      status.StatusNotOk if failed to get the embedding vector. Need to import
        the module to catch this error: `from pybind11_abseil import status`,
        see https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    return self._embedder.embed(audio.get_data())

  def cosine_similarity(self, u: embeddings_pb2.FeatureVector,
                        v: embeddings_pb2.FeatureVector) -> float:
    """Computes cosine similarity [1] between two feature vectors."""
    return self._embedder.cosine_similarity(u, v)

  def get_embedding_dimension(self, output_index: int) -> int:
    """Gets the dimensionality of the embedding output.
    Args:
      output_index: The output index of output layer.
    Returns:
      Dimensionality of the embedding output by the output_index'th output
      layer. Returns -1 if `output_index` is out of bounds.
    """
    return self._embedder.get_embedding_dimension(output_index)

  @property
  def number_of_output_layers(self) -> int:
    """Gets the number of output layers of the model."""
    return self._embedder.get_number_of_output_layers()

  @property
  def required_input_buffer_size(self) -> int:
    """Gets the required input buffer size for the model."""
    return self._embedder.get_required_input_buffer_size()

  @property
  def required_audio_format(self) -> _CppAudioFormat:
    """Gets the required audio format for the model."""
    return self._embedder.get_required_audio_format()
