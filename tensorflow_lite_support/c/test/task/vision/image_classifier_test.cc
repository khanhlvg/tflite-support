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

#include "tensorflow_lite_support/c/task/vision/image_classifier.h"

#include <string.h>

#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow_lite_support/c/common.h"
#include "tensorflow_lite_support/c/task/processor/classification_result.h"
#include "tensorflow_lite_support/c/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

namespace tflite {
namespace task {
namespace vision {
namespace {

using ::testing::HasSubstr;
using ::tflite::support::StatusOr;
using ::tflite::task::JoinPath;

constexpr char kTestDataDirectory[] =
    "/tensorflow_lite_support/cc/test/testdata/task/"
    "vision/";
// Quantized model.
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";

// Float model.
constexpr char kMobileNetFloatWithMetadata[] = 
    "mobilenet_v2_1.0_224.tflite";

StatusOr<ImageData> LoadImage(const char* image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

void VerifyCategoryApproximatelyEqual(const TfLiteCategory& actual,
                              const TfLiteCategory& expected) {
  const float kPrecision = 1e-6;
  EXPECT_EQ(actual.index, expected.index);
  EXPECT_EQ(actual.score, expected.score);
  if (actual.label && expected.label)
    EXPECT_EQ(strcmp(actual.label,expected.label), 0);
  if (actual.display_name && expected.display_name)
    EXPECT_EQ(strcmp(actual.display_name,expected.display_name), 0);
  EXPECT_NEAR(actual.score, expected.score, kPrecision);                                                
}

void PartiallyVerifyCategoriesForFloatModel(const TfLiteCategory* categories) {
  const int numCategoriesToTest = 3;
  TfLiteCategory firstCategory = {.index = 934, .score = 0.7399742, .label = "cheeseburger"};
  TfLiteCategory secondCategory = {.index = 925, .score = 0.026928535, .label = "guacamole"};
  TfLiteCategory thirdCategory = {.index = 932, .score = 0.025737215, .label = "bagel"};

  TfLiteCategory expectedCategories[] = {
                                          firstCategory, 
                                          secondCategory,
                                          thirdCategory
                                        };

  for (int i = 0; i < numCategoriesToTest; i++) 
    VerifyCategoryApproximatelyEqual(categories[i], expectedCategories[i]);                                                 
}

void PartiallyVerifyCategoriesForQuantizedModel(const TfLiteCategory* categories) {
  const int numCategoriesToTest = 3;
  TfLiteCategory firstCategory = {.index = 934, .score = 0.96484375, .label = "cheeseburger"};
  TfLiteCategory secondCategory = {.index = 948, .score = 0.0078125, .label = "mushroom"};
  TfLiteCategory thirdCategory = {.index = 924, .score = 0.00390625, .label = "plate"};
  TfLiteCategory expectedCategories[] = {
                                          firstCategory, 
                                          secondCategory,
                                          thirdCategory
  };

  for (int i = 0; i < numCategoriesToTest; i++) 
    VerifyCategoryApproximatelyEqual(categories[i], expectedCategories[i]);                          
                              
}

void VerifyClassificationsWithUnboundedMaxResults(const TfLiteClassifications& classifications,
                              const int expected_head_index,
                              const int expected_size) {
  EXPECT_EQ(classifications.head_index, expected_head_index);
  EXPECT_EQ(classifications.size, expected_size);
  EXPECT_NE(classifications.categories, nullptr);                                        
}

void VerfiyClassificationResult(TfLiteClassificationResult *classification_result) {
  ASSERT_NE(classification_result, nullptr);
  EXPECT_GE(classification_result->size, 1);
  EXPECT_NE(classification_result->classifications, nullptr);
}

void VerfiyClassificationResultForQuantizedModel(TfLiteClassificationResult *classification_result) {
  const int kNumCategories = 1001;
  VerfiyClassificationResult(classification_result);
  VerifyClassificationsWithUnboundedMaxResults(classification_result->classifications[0], 0, kNumCategories);
  PartiallyVerifyCategoriesForQuantizedModel(classification_result->classifications[0].categories);
}

void VerfiyClassificationResultForFloatModel(TfLiteClassificationResult *classification_result) {
  const int kNumCategories = 1001;
  VerfiyClassificationResult(classification_result);
  VerifyClassificationsWithUnboundedMaxResults(classification_result->classifications[0], 0, kNumCategories);
  PartiallyVerifyCategoriesForQuantizedModel(classification_result->classifications[0].categories);
}

class ImageClassifierFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(ImageClassifierFromOptionsTest, FailsWithNullOptionsAndError) {
  TfLiteSupportError* error = nullptr;
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(nullptr, &error);

  EXPECT_EQ(image_classifier, nullptr);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Expected non null options"));

  TfLiteSupportErrorDelete(error);
}

TEST_F(ImageClassifierFromOptionsTest, FailsWithMissingModelPath) {
  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, nullptr);
  EXPECT_EQ(image_classifier, nullptr);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);
}

TEST_F(ImageClassifierFromOptionsTest, FailsWithMissingModelPathAndError) {
  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();

  TfLiteSupportError* error = nullptr;
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, &error);

  EXPECT_EQ(image_classifier, nullptr);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("`base_options.model_file`"));

  TfLiteSupportErrorDelete(error);
}

TEST_F(ImageClassifierFromOptionsTest, SucceedsWithModelPath) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata);
  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, nullptr);

  EXPECT_NE(image_classifier, nullptr);
  TfLiteImageClassifierDelete(image_classifier);
}

TEST_F(ImageClassifierFromOptionsTest, SucceedsWithNumberOfThreadsAndError) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata);
  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();
  options.base_options.compute_settings.cpu_settings.num_threads = 3;

  TfLiteSupportError* error = nullptr;
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, &error);

  EXPECT_NE(image_classifier, nullptr);
  EXPECT_EQ(error, nullptr);

  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);
  if (error) TfLiteSupportErrorDelete(error);
}

TEST_F(ImageClassifierFromOptionsTest,
       FailsWithClassNameDenyListAndClassNameAllowListAndError) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata);

  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();

  char* label_denylist[9] = {(char*)"brambling"};
  options.classification_options.label_denylist.list = label_denylist;
  options.classification_options.label_denylist.length = 1;

  char* label_allowlist[12] = {(char*)"cheeseburger"};
  options.classification_options.label_allowlist.list = label_allowlist;
  options.classification_options.label_allowlist.length = 1;

  TfLiteSupportError* error = nullptr;
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, &error);

  EXPECT_EQ(image_classifier, nullptr);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("mutually exclusive options"));

  TfLiteSupportErrorDelete(error);
}

TEST(ImageClassifierNullClassifierClassifyTest,
     FailsWithNullImageClassifierAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(nullptr, nullptr, &error);

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

class ImageClassifierQuantizedModelClassifyTest : public tflite_shims::testing::Test {
 protected:
  void SetUp() override {
    std::string model_path =
        JoinPath("./" /*test src dir*/, kTestDataDirectory,
                 kMobileNetQuantizedWithMetadata);

    TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
    options.base_options.model_file.file_path = model_path.data();
    image_classifier = TfLiteImageClassifierFromOptions(&options, nullptr);
    ASSERT_NE(image_classifier, nullptr);
  }

  void TearDown() override { TfLiteImageClassifierDelete(image_classifier); }
  TfLiteImageClassifier* image_classifier;
};

TEST_F(ImageClassifierQuantizedModelClassifyTest, SucceedsWithImageData) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);
  
  VerfiyClassificationResultForQuantizedModel(classification_result);

  TfLiteClassificationResultDelete(classification_result);
}

class ImageClassifierFloatModelClassifyTest : public tflite_shims::testing::Test {
 protected:
  void SetUp() override {
    std::string model_path =
        JoinPath("./" /*test src dir*/, kTestDataDirectory,
                 kMobileNetQuantizedWithMetadata);

    TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
    options.base_options.model_file.file_path = model_path.data();
    image_classifier = TfLiteImageClassifierFromOptions(&options, nullptr);
    ASSERT_NE(image_classifier, nullptr);
  }

  void TearDown() override { TfLiteImageClassifierDelete(image_classifier); }
  TfLiteImageClassifier* image_classifier;
};

TEST_F(ImageClassifierFloatModelClassifyTest, SucceedsWithImageData) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);

  VerfiyClassificationResultForFloatModel(classification_result);
  
  TfLiteClassificationResultDelete(classification_result);
}

TEST_F(ImageClassifierQuantizedModelClassifyTest, FailsWithNullFrameBufferAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, nullptr, &error);

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

TEST_F(ImageClassifierQuantizedModelClassifyTest, FailsWithNullImageDataAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {.format = kRGB, .orientation = kTopLeft};

  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer, &error);

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

TEST_F(ImageClassifierQuantizedModelClassifyTest, SucceedsWithRoiWithinImageBounds) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteBoundingBox bounding_box = {
      .origin_x = 0, .origin_y = 0, .width = 100, .height = 100};
  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassifyWithRoi(image_classifier, &frame_buffer,
                                           &bounding_box, &error);

  ImageDataFree(&image_data);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_GE(classification_result->size, 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_GE(classification_result->classifications->size, 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
                   "bagel"),
            0);
  EXPECT_GE(classification_result->classifications->categories[0].score, 0.30);

  TfLiteClassificationResultDelete(classification_result);
}

TEST_F(ImageClassifierQuantizedModelClassifyTest, FailsWithRoiOutsideImageBoundsAndError) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteBoundingBox bounding_box = {
      .origin_x = 0, .origin_y = 0, .width = 250, .height = 250};
  TfLiteSupportError* error = nullptr;
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassifyWithRoi(image_classifier, &frame_buffer,
                                           &bounding_box, &error);

  ImageDataFree(&image_data);

  EXPECT_EQ(classification_result, nullptr);
  if (classification_result)
    TfLiteClassificationResultDelete(classification_result);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, kInvalidArgumentError);
  EXPECT_NE(error->message, nullptr);
  EXPECT_THAT(error->message, HasSubstr("Invalid crop coordinates"));

  TfLiteSupportErrorDelete(error);
}

TEST(ImageClassifierWithUserDefinedOptionsClassifyTest,
     SucceedsWithClassNameDenyList) {
  char* denylisted_label_name = (char*)"cheeseburger";
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata);

  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();

  char* label_denylist[12] = {denylisted_label_name};
  options.classification_options.label_denylist.list = label_denylist;
  options.classification_options.label_denylist.length = 1;

  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, nullptr);
  ASSERT_NE(image_classifier, nullptr);

  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

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

TEST(ImageClassifierWithUserDefinedOptionsClassifyTest,
     SucceedsWithClassNameAllowList) {
  char* allowlisted_label_name = (char*)"cheeseburger";
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata)
          .data();

  TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();

  char* label_allowlist[12] = {allowlisted_label_name};
  options.classification_options.label_allowlist.list = label_allowlist;
  options.classification_options.label_allowlist.length = 1;

  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options, nullptr);
  ASSERT_NE(image_classifier, nullptr);

  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

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
