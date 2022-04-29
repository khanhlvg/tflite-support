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
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/c/common.h"
#include "tensorflow_lite_support/c/task/processor/classification_result.h"
#include "tensorflow_lite_support/c/task/audio/core/audio_buffer.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/cc/task/audio/utils/wav_io.h"

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

namespace tflite {
namespace task {
namespace audio {
namespace {

using ::testing::HasSubstr;
using ::tflite::support::StatusOr;
using ::tflite::task::JoinPath;

constexpr char kTestDataDirectory[] =
    "/tensorflow_lite_support/cc/test/testdata/task/"
    "audio/";
// Quantized model.
constexpr char kYamNetAudioClassifierWithMetadata[] =
    "yamnet_audio_classifier_with_metadata.tflite";

StatusOr<TfLiteAudioBuffer> LoadAudioBufferFromFileNamed(
    const std::string wav_file, int buffer_size) {
  std::string contents = ReadFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, wav_file));

  uint32_t decoded_sample_count;
  uint16_t decoded_channel_count;
  uint32_t decoded_sample_rate;
  std::vector<float> wav_data;

  absl::Status read_audio_file_status =
 DecodeLin16WaveAsFloatVector(
      contents, &wav_data, &decoded_sample_count, &decoded_channel_count,
      &decoded_sample_rate);

  if (decoded_sample_count > buffer_size) {
    decoded_sample_count = buffer_size;
  }

  if (!read_audio_file_status.ok()) {
    return read_audio_file_status;
  }

  float* c_wav_data = (float*)malloc(sizeof(float) * wav_data.size());
  if (!c_wav_data) {
    exit(-1);
  }

  memcpy(c_wav_data, wav_data.data(), sizeof(float) * wav_data.size());

  TfLiteAudioBuffer audio_buffer = {.format = {.channels = decoded_channel_count, 
                     .sample_rate = static_cast<int>(decoded_sample_rate)}, 
          .data = c_wav_data, 
          .size = decoded_sample_count};

  return audio_buffer;
}

void Verify(TfLiteClassificationResult &classification_result, int expected_classifications_size) {
  EXPECT_NE(classification_result, NULL);
  EXPECT_EQ(classification_result.size, expected_classifications_size);
  EXPECT_NE(classification_result.classifications, NULL);
}

void Verify(TfLiteClassifications &classifications, int expected_categories_size, int expected_head_index, char* expected_head_name) {
  EXPECT_EQ(classifications.size, expected_categories_size);
  EXPECT_NE(classifications.head_index, expected_head_index);
  EXPECT_EQ(classifications.head_name, expected_head_name);
  EXPECT_NE(classification_result.categories, NULL);
}

void Verify(TfLiteCategory &category, int expected_index, char* expected_label, char* expected_display_name, float expected_score) {
  const float kPrecision = 1e-6;
  EXPECT_EQ(category.index, expected_index);
  EXPECT_EQ(category.label, expected_label);
  EXPECT_EQ(category.display_name, expected_display_name);
  EXPECT_NEAR(category.score, expected_score, kPrecision);
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
  EXPECT_THAT(error->message, HasSubstr("INVALID_ARGUMENT: Missing mandatory `model_file` field in `base_options`"));

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

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(nullptr, nullptr, &error);

  EXPECT_EQ(classification_result, nullptr);
  if (classification_result)
    TfLiteClassificationResultDelete(classification_result);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Expected non null audio classifier"));

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
  int input_buffer_size = TfLiteAudioClassifierGetRequiredInputBufferSize(audio_classifier, nullptr);
  ASSERT_NE(input_buffer_size, -1);

  SUPPORT_ASSERT_OK_AND_ASSIGN(TfLiteAudioBuffer audio_buffer, LoadAudioBufferFromFileNamed("speech.wav", input_buffer_size));

  TfLiteSupportError *classifyError = NULL;
  TfLiteClassificationResult* classification_result =
      TfLiteAudioClassifierClassify(audio_classifier, &audio_buffer, &classifyError);

  free((void *)(audio_buffer.data));

  ASSERT_NE(classification_result, nullptr);
  EXPECT_GE(classification_result->size, 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_GE(classification_result->classifications->size, 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  
  Verify(classification_result, 1);
  Verify(classification_result.classifications[0], 0, "scores");
  Verify(classification_result.classifications[0].categories[0], 0, , "Speech", NULL, 0.917969);
  Verify(classification_result.classifications[0].categories[1], 500, , "Inside, small room", NULL, 0.058594);
  Verify(classification_result.classifications[0].categories[2], 494, , "Silence", NULL, 0.011719);



//  classifications {
//   classes {
//     index: 0
//     score: 0.917969
//     class_name: "Speech"
//   }
//   classes {
//     index: 500
//     score: 0.058594
//     class_name: "Inside, small room"
//   }
//   classes {
//     index: 494
//     score: 0.011719
//     class_name: "Silence"
//   }
//   head_index: 0
//   head_name: "scores"
// }

  
  TfLiteClassificationResultDelete(classification_result);
}

// TEST_F(AudioClassifierClassifyTest, FailsWithNullFrameBufferAndError) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteAudioClassifierClassify(audio_classifier, nullptr, &error);

//   ImageDataFree(&image_data);

//   EXPECT_EQ(classification_result, nullptr);
//   if (classification_result)
//     TfLiteClassificationResultDelete(classification_result);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Expected non null frame buffer"));

//   TfLiteSupportErrorDelete(error);
// }

// TEST_F(AudioClassifierClassifyTest, FailsWithNullImageDataAndError) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {.format = kRGB, .orientation = kTopLeft};

//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, &error);

//   ImageDataFree(&image_data);

//   EXPECT_EQ(classification_result, nullptr);
//   if (classification_result)
//     TfLiteClassificationResultDelete(classification_result);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Invalid stride information"));

//   TfLiteSupportErrorDelete(error);
// }

// TEST(AudioClassifierWithUserDefinedOptionsClassifyTest,
//      SucceedsWithClassNameDenyList) {
//   char* denylisted_label_name = (char*)"cheeseburger";
//   std::string model_path =
//       JoinPath("./" /*test src dir*/, kTestDataDirectory,
//                kYamNetAudioClassifierWithMetadata);

//   TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
//   options.base_options.model_file.file_path = model_path.data();

//   char* label_denylist[12] = {denylisted_label_name};
//   options.classification_options.label_denylist.list = label_denylist;
//   options.classification_options.label_denylist.length = 1;

//   TfLiteAudioClassifier* audio_classifier =
//       TfLiteAudioClassifierFromOptions(&options, nullptr);
//   ASSERT_NE(audio_classifier, nullptr);

//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {
//       .format = kRGB,
//       .orientation = kTopLeft,
//       .dimension = {.width = image_data.width, .height = image_data.height},
//       .buffer = image_data.pixel_data};

//   TfLiteClassificationResult* classification_result =
//       TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, nullptr);

//   ImageDataFree(&image_data);
//   if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

//   ASSERT_NE(classification_result, nullptr);
//   EXPECT_GE(classification_result->size, 1);
//   EXPECT_NE(classification_result->classifications, nullptr);
//   EXPECT_GE(classification_result->classifications->size, 1);
//   EXPECT_NE(classification_result->classifications->categories, nullptr);
//   EXPECT_NE(strcmp(classification_result->classifications->categories[0].label,
//                    denylisted_label_name),
//             0);

//   TfLiteClassificationResultDelete(classification_result);
// }

// TEST(AudioClassifierWithUserDefinedOptionsClassifyTest,
//      SucceedsWithClassNameAllowList) {
//   char* allowlisted_label_name = (char*)"cheeseburger";
//   std::string model_path =
//       JoinPath("./" /*test src dir*/, kTestDataDirectory,
//                kYamNetAudioClassifierWithMetadata)
//           .data();

//   TfLiteAudioClassifierOptions options = TfLiteAudioClassifierOptionsCreate();
//   options.base_options.model_file.file_path = model_path.data();

//   char* label_allowlist[12] = {allowlisted_label_name};
//   options.classification_options.label_allowlist.list = label_allowlist;
//   options.classification_options.label_allowlist.length = 1;

//   TfLiteAudioClassifier* audio_classifier =
//       TfLiteAudioClassifierFromOptions(&options, nullptr);
//   ASSERT_NE(audio_classifier, nullptr);

//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadWavFile("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {
//       .format = kRGB,
//       .orientation = kTopLeft,
//       .dimension = {.width = image_data.width, .height = image_data.height},
//       .buffer = image_data.pixel_data};

//   TfLiteClassificationResult* classification_result =
//       TfLiteAudioClassifierClassify(audio_classifier, &frame_buffer, nullptr);

//   ImageDataFree(&image_data);
//   if (audio_classifier) TfLiteAudioClassifierDelete(audio_classifier);

//   ASSERT_NE(classification_result, nullptr);
//   EXPECT_GE(classification_result->size, 1);
//   EXPECT_NE(classification_result->classifications, nullptr);
//   EXPECT_GE(classification_result->classifications->size, 1);
//   EXPECT_NE(classification_result->classifications->categories, nullptr);
//   EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
//                    allowlisted_label_name),
//             0);

//   TfLiteClassificationResultDelete(classification_result);
// }

}  // namespace
}  // namespace audio
}  // namespace task
}  // namespace tflite
