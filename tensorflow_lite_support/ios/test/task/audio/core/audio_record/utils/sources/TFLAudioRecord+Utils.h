// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import "tensorflow_lite_support/ios/task/audio/core/audio_record/sources/TFLAudioRecord.h"

NS_ASSUME_NONNULL_BEGIN

// This category of TFLAudioRecord is private to the test files. This is needed in order to
// expose the method to load the audio record buffer without calling: -[TFLAudioRecord
// startRecordingWithError:]. This is needed to avoid exposing this method which isn't useful to the
// consumers of the framework.
@interface TFLAudioRecord (Tests)
- (void)mockLoadBufferWithFileName:(NSString *)fileName extension:(NSString *)extension;
@end

NS_ASSUME_NONNULL_END
