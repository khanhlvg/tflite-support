/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/** Holds a confidence mask belonging to a single class and its meta data. */
NS_SWIFT_NAME(NearestNeighbor)
@interface TFLNearestNeighbor : NSObject

/**
 * User-defined metadata about the result. This could be a label, a unique ID, a serialized proto of some sort, etc.
 */
@property(nonatomic, readonly) NSData *metadata;

/**
 * The distance score indicating how confident the result is. Lower is better.
 */
@property(nonatomic, readonly) CGFloat distance;

/**
 * Initializes a confidence mask.
 */
- (instancetype)initWithMetaData:(NSData *)metadata distance:(CGFloat)distance;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/** Holds category mask and its metadata. */
NS_SWIFT_NAME(SearchResult)
@interface TFLSearchResult : NSObject

/**
 * Flattened 2D-array of size `width` x `height`, in row major order.
 * The value of each pixel in this mask represents the class to which the
 * pixel belongs.
 */
@property(nonatomic, readonly) NSArray<TFLNearestNeighbor *> *nearestNeighbors;


+ (instancetype)new NS_UNAVAILABLE;

/**
 * Initializes a new `TFLCategoryMask` mask.
 *
 * @param nearestNeighbors Width of the mask.
 *
 * @return An instance of TFLSearchResult initialized to the specified values.
 */
- (instancetype)initWithNearestNeighbors:(NSArray<TFLNearestNeighbor *> *)nearestNeighbors;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
