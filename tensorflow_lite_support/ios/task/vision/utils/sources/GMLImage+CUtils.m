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
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+Utils.h"

#import <CoreGraphics/CoreGraphics.h>

@implementation GMLImage (CUtils)

- (nullable TfLiteFrameBuffer *)cFrameBufferWithError:(NSError *_Nullable *)error {
  TfLiteFrameBuffer *cFrameBuffer = NULL;

  uint8_t *buffer = [self bufferWithError:error];

  if (!buffer) {
    return NULL;
  }

  CGSize bitmapSize = self.bitmapSize;
  enum TfLiteFrameBufferFormat cFrameBufferFormat = kRGB;


  TfLiteFrameBuffer *cFrameBuffer = [TFLCommonUtils mallocWithSize:sizeof(TfLiteFrameBuffer)
                                                             error:error];

  if (cFrameBuffer) {
    cFrameBuffer->dimension.width = bitmapSize.width;
    cFrameBuffer->dimension.height = bitmapSize.height;
    cFrameBuffer->buffer = buffer;
    cFrameBuffer->format = frameBufferFormat;
  }

  return cFrameBuffer;
}

@end
