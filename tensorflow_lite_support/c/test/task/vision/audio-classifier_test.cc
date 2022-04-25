/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_lite_support/c/task/audio/audio_classifier.h"

#include <string.h>

#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow_lite_support/c/common.h"
#include "tensorflow_lite_support/c/task/processor/classification_result.h"
#include "tensorflow_lite_support/c/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/cc/task/audio/utils/wav_io.h"

namespace tflite {
namespace task {
namespace audio {
namespace {

using ::testing::HasSubstr;
using ::tflite::support::StatusOr;
using ::tflite::task::JoinPath;

constexpr char kTestDataDirectory[] =
    "/tensorflow_lite_support/cc/test/testdata/task/"
    "vision/";
// Quantized model.
constexpr char kYamNetAudioClassifierWithMetadata[] =
    "yamnet_audio_classifier_with_metadata.tflite";

StatusOr<ImageData> LoadWavFile(const char* file_name) {
  return DecodeLin16WaveAsFloatVector(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, file_name));
}

class AudioClassifierFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(AudioClassifierFromOptionsTest, FailsWithNullOptionsAndError) {
  TfLiteSupportError* error = nullptr;
  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(nullptr, &error);

  EXPECT_EQ(audio_classifier, nullptr);
  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Expected non null options"));

  TfLiteSupportErrorDelete(error);
}

TEST_F(AudioClassifierFromOptionsTest, FailsWithMissingModelPath) {
  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, nullptr);
  EXPECT_EQ(audio_classifier, nullptr);
  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);
}

TEST_F(AudioClassifierFromOptionsTest, FailsWithMissingModelPathAndError) {
  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();

  TfLiteSupportError* error = nullptr;
  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, &error);

  EXPECT_EQ(audio_classifier, nullptr);
  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("`base_options.model_file`"));

  TfLiteSupportErrorDelete(error);
}

TEST_F(AudioClassifierFromOptionsTest, SucceedsWithModelPath) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kYamNetAudioClassifierWithMetadata);
  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();
  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, nullptr);

  EXPECT_NE(audio_classifier, nullptr);
  TfLiteAudioClassifierDelete(audio_classifier);
}

TEST_F(AudioClassifierFromOptionsTest, SucceedsWithNumberOfThreadsAndError) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kYamNetAudioClassifierWithMetadata);
  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();
  options.base_options.compute_settings.cpu_settings.num_threads = 3;

  TfLiteSupportError* error = nullptr;
  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, &error);

  EXPECT_NE(audio_classifier, nullptr);
  EXPECT_EQ(error, nullptr);

  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);
  if (error) TfLiteSupportErrorDelete(error);
}

TEST_F(AudioClassifierFromOptionsTest,
       FailsWithClassNameDenyListAndClassNameAllowListAndError) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kYamNetAudioClassifierWithMetadata);

  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();

  char* label_denylist[9] = {(char*)"Speech"};
  options.classification_options.label_denylist.list = label_denylist;
  options.classification_options.label_denylist.length = 1;

  char* label_allowlist[12] = {(char*)"Silence"};
  options.classification_options.label_allowlist.list = label_allowlist;
  options.classification_options.label_allowlist.length = 1;

  TfLiteSupportError* error = nullptr;
  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, &error);

  EXPECT_EQ(audio_classifier, nullptr);
  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("mutually exclusive options"));

  TfLiteSupportErrorDelete(error);
}

TEST(AudioClassifierNullClassifierClassifyTest,
     FailsWithNullAudioClassifierAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(nullptr, nullptr, &error);

  ImageDataFree(&image_data);

  EXPECT_EQ(classification_result, nullptr);
  if (classification_result)
    TfLiteClassificationResultDelete(classification_result);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Expected non null image classifier"));

  TfLiteSupportErrorDelete(error);
}

class AudioClassifierClassifyTest : public tflite_shims::testing::Test {
 protected:
  void SetUp() override {
    std::string model_path =
        JoinPath("./" /*test src dir*/, kTestDataDirectory,
                 kYamNetAudioClassifierWithMetadata);

    TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
    options.base_options.model_file.file_path = model_path.data();
    audio_classifier = TfLiteAudioClassifierFromOptions(&options, nullptr);
    ASSERT_NE(audio_classifier, nullptr);
  }

  void TearDown() override { TfLiteAudioClassifierDelete(audio_classifier); }
  TfLiteAudioClassifier* audio_classifier;
};

TEST_F(AudioClassifierClassifyTest, SucceedsWithImageData) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_GE(classification_result->size, 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_GE(classification_result->classifications->size, 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
                   "cheeseburger"),
            0);
  EXPECT_GE(classification_result->classifications->categories[0].score, 0.90);

  TfLiteClassificationResultDelete(classification_result);
}

TEST_F(AudioClassifierClassifyTest, FailsWithNullFrameBufferAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(audio_classifier, nullptr, &error);

  ImageDataFree(&image_data);

  EXPECT_EQ(classification_result, nullptr);
  if (classification_result)
    TfLiteClassificationResultDelete(classification_result);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Expected non null frame buffer"));

  TfLiteSupportErrorDelete(error);
}

TEST_F(AudioClassifierClassifyTest, FailsWithNullImageDataAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {.format = kRGB, .orientation = kTopLeft};

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, &error);

  ImageDataFree(&image_data);

  EXPECT_EQ(classification_result, nullptr);
  if (classification_result)
    TfLiteClassificationResultDelete(classification_result);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Invalid stride information"));

  TfLiteSupportErrorDelete(error);
}

TEST(AudioClassifierWithUserDefinedOptionsClassifyTest,
     SucceedsWithClassNameDenyList) {
  char* denylisted_label_name = (char*)"cheeseburger";
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kYamNetAudioClassifierWithMetadata);

  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();

  char* label_denylist[12] = {denylisted_label_name};
  options.classification_options.label_denylist.list = label_denylist;
  options.classification_options.label_denylist.length = 1;

  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, nullptr);
  ASSERT_NE(audio_classifier, nullptr);

  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);
  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_GE(classification_result->size, 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_GE(classification_result->classifications->size, 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  EXPECT_NE(strcmp(classification_result->classifications->categories[0].label,
                   denylisted_label_name),
            0);

  TfLiteClassificationResultDelete(classification_result);
}

TEST(AudioClassifierWithUserDefinedOptionsClassifyTest,
     SucceedsWithClassNameAllowList) {
  char* allowlisted_label_name = (char*)"cheeseburger";
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kYamNetAudioClassifierWithMetadata)
          .data();

  TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();

  char* label_allowlist[12] = {allowlisted_label_name};
  options.classification_options.label_allowlist.list = label_allowlist;
  options.classification_options.label_allowlist.length = 1;

  TfLiteAudioClassifier* audio_classifier =
      TfLiteAudioClassifierFromOptions(&options, nullptr);
  ASSERT_NE(audio_classifier, nullptr);

  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);
  if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_GE(classification_result->size, 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_GE(classification_result->classifications->size, 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
                   allowlisted_label_name),
            0);

  TfLiteClassificationResultDelete(classification_result);
}

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
