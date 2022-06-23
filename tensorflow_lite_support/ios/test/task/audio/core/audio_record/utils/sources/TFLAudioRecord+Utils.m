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

#import "tensorflow_lite_support/ios/test/task/audio/core/audio_record/utils/sources/TFLAudioRecord+Utils.h"

@implementation TFLAudioRecord (Utils)

- (void)mockLoadBufferWithFileName:(NSString *)fileName extension:(NSString *)extension {
  AVAudioNode *inputNode = [_audioEngine inputNode];
  AVAudioFormat *format = [inputNode outputFormatForBus:0];
  // Loading AVAudioPCMBuffer with an array is not currently supported for iOS versions < 15.0.
  // Instead audio samples from a wav file are loaded and converted into the same format
  // of AVAudioEngine's input node to mock the input from the AVAudio Engine.
  AVAudioPCMBuffer *audioEngineBuffer = [self bufferFromFileWithName:fileName
                                                           extension:extension
                                                    processingFormat:format];
  XCTAssertNotNil(audioEngineBuffer);

  // Convert the buffer in the audio engine input format to the format with which audio record is
  // intended to output the audio samples. This mocks the internal conversion of audio record when
  // -[TFLAudioRecord startRecording:withError:] is called.
  AVAudioFormat *recordingFormat = [[AVAudioFormat alloc]
      initWithCommonFormat:AVAudioPCMFormatFloat32
                sampleRate:self.audioFormat.sampleRate
                  channels:(AVAudioChannelCount)self.audioFormat.channelCount
               interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:self.audioEngineFormat
                                                                     toFormat:recordingFormat];
  // Convert and load the buffer of `TFLAudioRecord`.
  [self convertAndLoadBuffer:audioEngineBuffer usingAudioConverter:audioConverter];
}

@end
