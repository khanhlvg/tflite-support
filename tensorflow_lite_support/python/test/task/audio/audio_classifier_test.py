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
"""Tests for audio_classifier."""

import enum
import json

from absl.testing import parameterized
from google.protobuf import json_format
# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.
import unittest
from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.audio import audio_classifier
from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util

_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_AudioClassifier = audio_classifier.AudioClassifier
_AudioClassifierOptions = audio_classifier.AudioClassifierOptions

_MODEL_FILE = 'yamnet_classification.tflite'
_AUDIO_FILE = 'meow_16k.wav'
_ACCEPTABLE_ERROR_RANGE = 0.000001


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class AudioClassifierTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.test_image_path = test_util.get_test_data_path(_AUDIO_FILE)
    self.model_path = test_util.get_test_data_path(_MODEL_FILE)

  @staticmethod
  def create_classifier_from_options(model_file, **classification_options):
    base_options = _BaseOptions(model_file=model_file)
    classification_options = classification_options_pb2.ClassificationOptions(
        **classification_options)
    options = _AudioClassifierOptions(
        base_options=base_options,
        classification_options=classification_options)
    classifier = _AudioClassifier.create_from_options(options)
    return classifier

  @staticmethod
  def build_test_data(expected_categories):
    classifications = classifications_pb2.Classifications(head_index=0)
    classifications.classes.extend(
        [class_pb2.Category(**args) for args in expected_categories])
    expected_result = classifications_pb2.ClassificationResult()
    expected_result.classifications.append(classifications)
    expected_result_dict = json.loads(
        json_format.MessageToJson(expected_result))

    return expected_result_dict

  def test_classify_model(self):
    # Creates classifier.
    model_file = _ExternalFile(file_name=self.model_path)

    classifier = self.create_classifier_from_options(
        model_file, max_results=3)

    input_buffer_size = classifier.required_input_buffer_size
    audio_format = classifier.required_audio_format

    print("Fetching required input buffer size, channels & sample rate")
    print(input_buffer_size, audio_format.channels, audio_format.sample_rate)

    # Loads audio.
    audio = tensor_audio.TensorAudio.from_file(
      self.test_image_path, input_buffer_size)

    print("Fetching input audio info")
    input_sample_count = audio.audio_data.sample_count
    input_channels = audio.audio_data.channels
    input_sample_rate = audio.audio_data.sample_rate
    print(input_sample_count, input_channels, input_sample_rate)

    # Ensure that the WAV file's sampling rate matches with the model
    # requirement.
    self.assertEqual(
      input_sample_rate, audio_format.sample_rate,
      'The test audio\'s sample rate does not match with the model\'s '
      'requirement.'
    )

    # Classifies the input.
    # audio_result = classifier.classify(audio)
    # audio_result_dict = json.loads(json_format.MessageToJson(audio_result))
    # print(audio_result_dict)


if __name__ == '__main__':
  unittest.main()
