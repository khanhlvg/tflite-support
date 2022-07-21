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
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+Utils.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+CppUtils.h"

#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"

#import <CoreGraphics/CoreGraphics.h>

namespace {
using FrameBufferCpp = ::tflite::task::vision::FrameBuffer;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
}  // namespace

@implementation GMLImage (CppUtils)

- (StatusOr<std::unique_ptr<FrameBufferCpp>>)cppFrameBufferWithError:(NSError *_Nullable *)error {
  uint8_t *buffer = [self bufferWithError:error];

  if (!buffer) {
    return NULL;
  }

  TfLiteFrameBuffer *cFrameBuffer = NULL;

  CGSize bitmapSize = self.bitmapSize;
  FrameBufferCpp::Format frame_buffer_format = FrameBufferCpp::Format::kRGB;

 StatusOr<std::unique_ptr<FrameBufferCpp>> frameBuffer = CreateFromRawBuffer(
      buffer,
      {(int)bitmapSize.width, (int)bitmapSize.height},
      frame_buffer_format, FrameBufferCpp::Orientation::kTopLeft);

  if (frameBuffer.status() != ok) {
    return NULL;
  }

  return frameBuffer.value();
}

@end
