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
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util


_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_AudioClassifier = audio_classifier.AudioClassifier
_AudioClassifierOptions = audio_classifier.AudioClassifierOptions

_FIXED_INPUT_SIZE_MODEL_FILE = 'yamnet_audio_classifier_with_metadata.tflite'
_SPEECH_AUDIO_FILE = 'speech.wav'
_FIXED_INPUT_SIZE_MODEL_CLASSIFICATIONS = {
  'scores': [
    {
      'index': 0,
      'score': 0.91796875,
      'class_name': 'Speech'}
    ,
    {
      'index': 500,
      'score': 0.05859375,
      'class_name': 'Inside, small room'
    },
    {
      'index': 494,
      'score': 0.015625,
      'class_name': 'Silence'
    }
  ]
}

_MULTIHEAD_MODEL_FILE = 'two_heads.tflite'
_TWO_HEADS_AUDIO_FILE = 'two_heads.wav'
_MULTIHEAD_MODEL_CLASSIFICATIONS = {
  'yamnet_classification': [
    {
      'index': 508,
      'score': 0.5486158,
      'class_name': 'Environmental noise'
    },
    {
      'index': 507,
      'score': 0.38086897,
      'class_name': 'Noise'
    },
    {
      'index': 106,
      'score': 0.25613675,
      'class_name': 'Bird'
    }
  ],
  'bird_classification': [
    {
      'index': 4,
      'score': 0.93399656,
      'class_name': 'Chestnut-crowned Antpitta'
    },
    {
      'index': 1,
      'score': 0.065934494,
      'class_name': 'White-breasted Wood-Wren'},
    {
      'index': 0,
      'score': 6.1469495e-05,
      'class_name': 'Red Crossbill'
    }
  ]
}

_ALLOW_LIST = ['Speech', 'Inside, small room']
_DENY_LIST = ['Speech']
_SCORE_THRESHOLD = 0.5
_MAX_RESULTS = 3
_ACCEPTABLE_ERROR_RANGE = 0.000001


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class AudioClassifierTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.test_audio_path = test_util.get_test_data_path(_SPEECH_AUDIO_FILE)
    self.model_path = test_util.get_test_data_path(_FIXED_INPUT_SIZE_MODEL_FILE)

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
  def build_test_data(classifications):
    expected_result = classifications_pb2.ClassificationResult()

    for index, (head_name, categories) in enumerate(classifications.items()):
      classifications = classifications_pb2.Classifications(
        head_index=index, head_name=head_name)
      classifications.classes.extend(
          [class_pb2.Category(**args) for args in categories])
      expected_result.classifications.append(classifications)

    expected_result_dict = json.loads(
        json_format.MessageToJson(expected_result))

    return expected_result_dict

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(
      model_file=_ExternalFile(file_name=self.model_path))
    options = _AudioClassifierOptions(base_options=base_options)
    classifier = _AudioClassifier.create_from_options(options)
    self.assertIsInstance(classifier, _AudioClassifier)

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
      options = _AudioClassifierOptions(base_options=base_options)
      _AudioClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(
        model_file=_ExternalFile(file_content=f.read()))
      options = _AudioClassifierOptions(base_options=base_options)
      classifier = _AudioClassifier.create_from_options(options)
      self.assertIsInstance(classifier, _AudioClassifier)

  @parameterized.parameters(
    (_FIXED_INPUT_SIZE_MODEL_FILE, ModelFileType.FILE_NAME,
     _SPEECH_AUDIO_FILE, 3, _FIXED_INPUT_SIZE_MODEL_CLASSIFICATIONS),
    (_FIXED_INPUT_SIZE_MODEL_FILE, ModelFileType.FILE_CONTENT,
     _SPEECH_AUDIO_FILE, 3, _FIXED_INPUT_SIZE_MODEL_CLASSIFICATIONS),
  )
  def test_classify_fixed_input_size_model(
      self, model_name, model_file_type, audio_file_name, max_results,
      expected_classifications
  ):
    # Get model path.
    model_path = test_util.get_test_data_path(model_name)

    # Creates classifier.
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    classifier = self.create_classifier_from_options(
        model_file, max_results=max_results)
    tensor = classifier.create_input_tensor_audio()

    # Load the input audio file.
    test_audio_path = test_util.get_test_data_path(audio_file_name)
    tensor.load_from_file(test_audio_path)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_classifications)

    # Comparing results.
    self.assertDeepAlmostEqual(
      audio_result_dict, expected_result_dict, delta=_ACCEPTABLE_ERROR_RANGE)

  @parameterized.parameters(
    (_MULTIHEAD_MODEL_FILE, ModelFileType.FILE_NAME, _TWO_HEADS_AUDIO_FILE,
     3, _MULTIHEAD_MODEL_CLASSIFICATIONS),
    (_MULTIHEAD_MODEL_FILE, ModelFileType.FILE_CONTENT, _TWO_HEADS_AUDIO_FILE,
     3, _MULTIHEAD_MODEL_CLASSIFICATIONS)
  )
  def test_classify_multi_head_model(
      self, model_name, model_file_type, audio_file_name, max_results,
      expected_classifications):
    # Get model path.
    model_path = test_util.get_test_data_path(model_name)

    # Creates classifier.
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    classifier = self.create_classifier_from_options(
      model_file, max_results=max_results)
    tensor = classifier.create_input_tensor_audio()

    # Load the input audio file.
    test_audio_path = test_util.get_test_data_path(audio_file_name)
    tensor.load_from_file(test_audio_path)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_classifications)

    # Comparing results.
    self.assertDeepAlmostEqual(
      audio_result_dict, expected_result_dict,
      delta=_ACCEPTABLE_ERROR_RANGE)

  def test_max_results_option(self):
    # Creates classifier.
    model_file = _ExternalFile(file_name=self.model_path)

    classifier = self.create_classifier_from_options(
      model_file, max_results=_MAX_RESULTS)
    tensor = classifier.create_input_tensor_audio()

    # Load the input audio file.
    tensor.load_from_file(self.test_audio_path)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    categories = audio_result_dict['classifications'][0]['classes']

    self.assertLessEqual(
      len(categories), _MAX_RESULTS, 'Too many results returned.')

  def test_score_threshold_option(self):
    # Creates classifier.
    model_file = _ExternalFile(file_name=self.model_path)

    classifier = self.create_classifier_from_options(
      model_file, score_threshold=_SCORE_THRESHOLD)
    tensor = classifier.create_input_tensor_audio()

    # Load the input audio file.
    tensor.load_from_file(self.test_audio_path)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    categories = audio_result_dict['classifications'][0]['classes']

    for category in categories:
      score = category['score']
      self.assertGreaterEqual(
        score, _SCORE_THRESHOLD,
        'Classification with score lower than threshold found. {0}'.format(
          category))

  def test_allowlist_option(self):
    # Creates classifier.
    model_file = _ExternalFile(file_name=self.model_path)

    classifier = self.create_classifier_from_options(
      model_file, class_name_allowlist=_ALLOW_LIST)
    tensor = classifier.create_input_tensor_audio()

    # Load the input audio file.
    tensor.load_from_file(self.test_audio_path)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    categories = audio_result_dict['classifications'][0]['classes']

    for category in categories:
      label = category['className']
      self.assertIn(
        label, _ALLOW_LIST,
        'Label "{0}" found but not in label allow list'.format(label))

  def test_denylist_option(self):
    # Creates classifier.
    model_file = _ExternalFile(file_name=self.model_path)

    classifier = self.create_classifier_from_options(
      model_file, score_threshold=0.01, class_name_denylist=_DENY_LIST)
    tensor = classifier.create_input_tensor_audio()

    # Load the input audio file.
    tensor.load_from_file(self.test_audio_path)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    categories = audio_result_dict['classifications'][0]['classes']

    for category in categories:
      label = category['className']
      self.assertNotIn(label, _DENY_LIST,
                       'Label "{0}" found but in deny list.'.format(label))

  def test_combined_allowlist_and_denylist(self):
    # Fails with combined allowlist and denylist
    with self.assertRaisesRegex(
        Exception,
        r'INVALID_ARGUMENT: `class_name_allowlist` and `class_name_denylist` '
        r'are mutually exclusive options. '
        r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = _BaseOptions(
        model_file=_ExternalFile(file_name=self.model_path))
      classification_options = classification_options_pb2.ClassificationOptions(
        class_name_allowlist=['foo'], class_name_denylist=['bar'])
      options = _AudioClassifierOptions(
        base_options=base_options,
        classification_options=classification_options)
      _AudioClassifier.create_from_options(options)

  def test_equal(self):
    base_options1 = _BaseOptions(
      model_file=_ExternalFile(file_name=self.model_path))
    options1 = _AudioClassifierOptions(base_options=base_options1)
    classifier1 = _AudioClassifier.create_from_options(options1)
    # Checks the same classifier object.
    self.assertEqual(classifier1, classifier1)

    base_options2 = _BaseOptions(
      model_file=_ExternalFile(file_name=self.model_path))
    options2 = _AudioClassifierOptions(base_options=base_options2)
    classifier2 = _AudioClassifier.create_from_options(options2)
    # Checks the classifiers with same file name.
    self.assertEqual(classifier1, classifier2)

    with open(self.model_path, 'rb') as f:
      model_content = f.read()
    base_options3 = _BaseOptions(
      model_file=_ExternalFile(file_content=model_content))
    options3 = _AudioClassifierOptions(base_options=base_options3)
    classifier3 = _AudioClassifier.create_from_options(options3)
    # Checks one classifier with file_name and the other with model_content.
    self.assertNotEqual(classifier1, classifier3)

    base_options4 = _BaseOptions(
      model_file=_ExternalFile(file_name=self.model_path))
    options4 = _AudioClassifierOptions(base_options=base_options4)
    options4.classification_options = classification_options_pb2.ClassificationOptions(
      score_threshold=0.5)
    classifier4 = _AudioClassifier.create_from_options(options4)
    # Checks the classifiers with different classification options.
    self.assertNotEqual(classifier1, classifier4)


if __name__ == '__main__':
  unittest.main()
