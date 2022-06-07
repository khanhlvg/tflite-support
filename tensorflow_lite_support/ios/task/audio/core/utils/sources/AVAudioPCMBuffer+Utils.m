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

#import "tensorflow_lite_support/ios/test/task/audio/core/audio_record/utils/sources/AVAudioPCMBuffer+Utils.h"

@implementation AVAudioPCMBuffer (Utils)

- (AVAudioPCMBuffer *)bufferUsingAudioConverter:(AVAudioConverter *)audioConverter error:(NSError **)error {
  // Capacity of converted PCM buffer is calculated in order to maintain the same
  // latency as the input pcmBuffer.
  AVAudioFrameCount capacity = ceil(self.frameLength * audioConverter.outputFormat.sampleRate /
                                    audioConverter.inputFormat.sampleRate);
  AVAudioPCMBuffer *outPCMBuffer = [[AVAudioPCMBuffer alloc]
      initWithPCMFormat:audioConverter.outputFormat
          frameCapacity:capacity * (AVAudioFrameCount)audioConverter.outputFormat.channelCount];

  AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer *_Nullable(
      AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus *_Nonnull outStatus) {
    *outStatus = AVAudioConverterInputStatus_HaveData;
    return self;
  };

  NSError *conversionError = nil;
  AVAudioConverterOutputStatus converterStatus = [audioConverter convertToBuffer:outPCMBuffer
                                                                           error:&conversionError
                                                              withInputFromBlock:inputBlock];


  AVAudioConverterOutputStatus converterStatus = [audioConverter convertToBuffer:outPCMBuffer
                                                                           error:&conversionError
                                                              withInputFromBlock:inputBlock];
  switch (converterStatus) {
    case AVAudioConverterOutputStatus_HaveData: {
      return outPCMBuffer;
    }
    case AVAudioConverterOutputStatus_InputRanDry: {
      [TFLCommonUtils createCustomError:error
                             withDomain:TFLAudioRecordErrorDomain
                                   code:TFLAudioRecordErrorCodeProcessingError
                            description:@"Not enough input is available to satisfy the request."];
      break;
    }
    case AVAudioConverterOutputStatus_EndOfStream: {
      [TFLCommonUtils createCustomError:error
                             withDomain:TFLAudioRecordErrorDomain
                                   code:TFLAudioRecordErrorCodeProcessingError
                            description:@"Reached end of input audio stream."];
      break;
    }
    case AVAudioConverterOutputStatus_Error: {
      // Conversion failed so returning a nil. Reason of the error isn't important to the library's
      // users.
      NSString *errorDescription = conversionError.localizedDescription
                                       ? conversionError.localizedDescription
                                       : @"Some error occured while processing incoming audio "
                                         @"frames.";
      [TFLCommonUtils createCustomError:error
                             withDomain:TFLAudioRecordErrorDomain
                                   code:TFLAudioRecordErrorCodeProcessingError
                            description:errorDescription];
      break;
    }
  }

  return nil;
}

- (nullable AVAudioPCMBuffer *)convertToAudioFormat:(AVAudioFormat *)audioFormat error:(NSError **)nil {
    AVAudioConverter *audioEngineConverter =
      [[AVAudioConverter alloc] initFromFormat:self.format toFormat:audioFormat];
  AVAudioPCMBuffer *audioEngineBuffer =
      [self bufferUsingAudioConverter:audioEngineConverter error:error];
  return audioEngineBuffer;
}

+ (nullable AVAudioPCMBuffer *)loadPCMBufferFromFileWithURL:(NSURL *)url processingFormat:(AVAudioFormat *)processingFormat error:(NSError **)error {

  AVAudioFile *audioFile = [[AVAudioFile alloc] initForReading:url error:error];

  if (!audioFile) {
    return nil;
  }

  AVAudioPCMBuffer *buffer =
      [[AVAudioPCMBuffer alloc] initWithPCMFormat:audioFile.processingFormat
                                    frameCapacity:(AVAudioFrameCount)audioFile.length];

  if([audioFile readIntoBuffer:buffer error:error]) {
    if ([processingFormat isEqual:buffer.format]) {
      return buffer;
    }
    else {
    return [buffer convertToAudioFormat:processingFormat error:error];
    }
  }

  return nil;
}
@end
