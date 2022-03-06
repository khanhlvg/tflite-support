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
from typing import Any, Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.core import task_utils
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_utils
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_buffer
from tensorflow_lite_support.python.task.audio.pybinds import _pywrap_audio_classifier
from tensorflow_lite_support.python.task.audio.pybinds import audio_classifier_options_pb2

_ProtoAudioClassifierOptions = audio_classifier_options_pb2.AudioClassifierOptions
_CppAudioClassifier = _pywrap_audio_classifier.AudioClassifier


@dataclasses.dataclass
class AudioClassifierOptions:
  """Options for the audio classifier task."""
  base_options: task_options.BaseOptions
  classification_options: Optional[
      classification_options_pb2.ClassificationOptions] = None

  def __eq__(self, other: Any) -> bool:
    if (not isinstance(other, self.__class__) or
        self.base_options != other.base_options):
      return False

    if self.classification_options is None and other.classification_options is None:
      return True
    elif (self.classification_options and other.classification_options and
          self.classification_options.SerializeToString()
          == self.classification_options.SerializeToString()):
      return True
    else:
      return False


class AudioClassifier(object):
  """Class that performs classification on audio."""

  def __init__(self, options: AudioClassifierOptions,
               classifier: _CppAudioClassifier) -> None:
    """Initializes the `AudioClassifier` object."""
    # Creates the object of C++ AudioClassifier class.
    self._options = options
    self._classifier = classifier

  @classmethod
  def create_from_options(cls,
                          options: AudioClassifierOptions) -> "AudioClassifier":
    """Creates the `AudioClassifier` object from audio classifier options.

    Args:
      options: Options for the audio classifier task.
    Returns:
      `AudioClassifier` object that's created from `options`.
    Raises:
      status.StatusNotOk if failed to create `AudioClassifier` object from
        `AudioClassifierOptions` such as missing the model. Need to import the
        module to catch this error: `from pybind11_abseil
        import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    # Creates the object of C++ AudioClassifier class.
    proto_options = _ProtoAudioClassifierOptions()
    proto_options.base_options.CopyFrom(
        task_utils.ConvertToProtoBaseOptions(options.base_options))

    # Updates values from classification_options.
    if options.classification_options:
      if options.classification_options.display_names_locale:
        proto_options.display_names_locale = \
          options.classification_options.display_names_locale
      if options.classification_options.max_results:
        proto_options.max_results = options.classification_options.max_results
      if options.classification_options.score_threshold:
        proto_options.score_threshold = \
          options.classification_options.score_threshold
      if options.classification_options.class_name_allowlist:
        proto_options.class_name_whitelist.extend(
            options.classification_options.class_name_allowlist)
      if options.classification_options.class_name_denylist:
        proto_options.class_name_blacklist.extend(
            options.classification_options.class_name_denylist)

    classifier = _CppAudioClassifier.create_from_options(proto_options)

    return cls(options, classifier)

  def classify(
      self,
      audio: tensor_audio.TensorAudio,
  ) -> classifications_pb2.ClassificationResult:
    """Performs classification on the provided TensorAudio.

    Args:
      audio: Tensor audio, used to extract the feature vectors.
    Returns:
      classification result.
    Raises:
      status.StatusNotOk if failed to get the feature vector. Need to import the
        module to catch this error: `from pybind11_abseil
        import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    audio_data = audio_utils.AudioData(*audio.get_data())
    return self._classifier.classify(audio_data)

  @property
  def required_input_buffer_size(self) -> int:
    """Gets the required input buffer size for the model."""
    return self._classifier.get_required_input_buffer_size()

  @property
  def required_audio_format(self) -> audio_buffer.AudioFormat:
    """Gets the required audio format for the model."""
    return self._classifier.get_required_audio_format()

  def __eq__(self, other: Any) -> bool:
    return (isinstance(other, self.__class__) and
            self._options == other._options)

  @property
  def options(self) -> AudioClassifierOptions:
    return self._options
