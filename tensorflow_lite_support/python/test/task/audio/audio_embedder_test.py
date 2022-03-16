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

from absl.testing import parameterized
# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.
import unittest

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.audio import audio_embedder
from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util


_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_AudioEmbedder = audio_embedder.AudioEmbedder
_AudioEmbedderOptions = audio_embedder.AudioEmbedderOptions

_YAMNET_EMBEDDING_MODEL_FILE = 'yamnet_audio_classifier_with_metadata.tflite'


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class AudioEmbedderTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_YAMNET_EMBEDDING_MODEL_FILE)

  @staticmethod
  def create_embedder_from_options(model_file, **embedding_options):
    base_options = _BaseOptions(model_file=model_file)
    embedding_options = embedding_options_pb2.EmbeddingOptions(
        **embedding_options)
    options = _AudioEmbedderOptions(
        base_options=base_options,
        embedding_options=embedding_options)
    embedder = _AudioEmbedder.create_from_options(options)
    return embedder

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    embedder = _AudioEmbedder.create_from_file(self.model_path)
    self.assertIsInstance(embedder, _AudioEmbedder)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    options = _AudioEmbedderOptions(
      _BaseOptions(model_file=_ExternalFile(file_name=self.model_path)))
    embedder = _AudioEmbedder.create_from_options(options)
    self.assertIsInstance(embedder, _AudioEmbedder)

  def test_create_from_options_fails_with_missing_model_file(self):
    # Missing the model file.
    with self.assertRaisesRegex(
        TypeError,
        r"__init__\(\) missing 1 required positional argument: 'model_file'"):
      _BaseOptions()

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        Exception,
        r'INVALID_ARGUMENT: Missing mandatory `model_file` field in '
        r'`base_options` '
        r"\[tflite::support::TfLiteSupportStatus='2']"):
      base_options = _BaseOptions(model_file=_ExternalFile(file_name=''))
      options = _AudioEmbedderOptions(base_options=base_options)
      _AudioEmbedder.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(
        model_file=_ExternalFile(file_content=f.read()))
      options = _AudioEmbedderOptions(base_options=base_options)
      embedder = _AudioEmbedder.create_from_options(options)
      self.assertIsInstance(embedder, _AudioEmbedder)

  @parameterized.parameters(
    (_YAMNET_EMBEDDING_MODEL_FILE, False, False, ModelFileType.FILE_NAME, 521,
     1),
    (_YAMNET_EMBEDDING_MODEL_FILE, True, True, ModelFileType.FILE_CONTENT, 521,
     1)
  )
  def test_embed(self, model_name, l2_normalize, quantize, model_file_type,
                 embedding_length, expected_similarity):
    # Create embedder.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    options = _AudioEmbedderOptions(
      _BaseOptions(model_file),
      embedding_options_pb2.EmbeddingOptions(
        l2_normalize=l2_normalize, quantize=quantize))
    embedder = _AudioEmbedder.create_from_options(options)

    # Load the input audio files.
    tensor0 = tensor_audio.TensorAudio.from_wav_file(
      test_util.get_test_data_path("speech.wav"),
      embedder.required_input_buffer_size)

    tensor1 = tensor_audio.TensorAudio.from_wav_file(
      test_util.get_test_data_path("speech.wav"),
      embedder.required_input_buffer_size)

    # Extract embeddings.
    result0 = embedder.embed(tensor0)
    result1 = embedder.embed(tensor1)

    # Check embedding sizes.
    def _check_embedding_size(result):
      self.assertLen(result.embeddings, 1)
      feature_vector = result.embeddings[0].feature_vector
      if quantize:
        self.assertLen(feature_vector.value_string, embedding_length)
      else:
        self.assertLen(feature_vector.value_float, embedding_length)

    _check_embedding_size(result0)
    _check_embedding_size(result1)

    result0_feature_vector = result0.embeddings[0].feature_vector
    result1_feature_vector = result1.embeddings[0].feature_vector

    if quantize:
      self.assertLen(result0_feature_vector.value_string, 521)
      self.assertLen(result1_feature_vector.value_string, 521)
    else:
      self.assertLen(result0_feature_vector.value_float, 521)
      self.assertLen(result1_feature_vector.value_float, 521)

    # Checks cosine similarity.
    similarity = embedder.cosine_similarity(
      result0_feature_vector, result1_feature_vector)
    self.assertAlmostEqual(similarity, expected_similarity, places=6)

  def test_get_embedding_dimension(self):
    options = _AudioEmbedderOptions(
      _BaseOptions(model_file=_ExternalFile(file_name=self.model_path)))
    embedder = _AudioEmbedder.create_from_options(options)
    self.assertEqual(embedder.get_embedding_dimension(0), 521)
    self.assertEqual(embedder.get_embedding_dimension(1), -1)

  def test_number_of_output_layers(self):
    options = _AudioEmbedderOptions(
      _BaseOptions(model_file=_ExternalFile(file_name=self.model_path)))
    embedder = _AudioEmbedder.create_from_options(options)
    self.assertEqual(embedder.number_of_output_layers, 1)


if __name__ == '__main__':
  unittest.main()
