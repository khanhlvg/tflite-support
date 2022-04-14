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
#import <CoreGraphics/CoreGraphics.h>
#import <XCTest/XCTest.h>

#import "tensorflow_lite_support/ios/task/vision/sources/TFLObjectDetector.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+Utils.h"

#define VerifyError(error, expectedDomain, expectedCode, expectedLocalizedDescription)  \
  XCTAssertNotNil(error);                                                               \
  XCTAssertEqual(error.domain, expectedDomain);                                         \
  XCTAssertEqual(error.code, expectedCode);                                             \
  XCTAssertNotEqual(                                                                    \
      [error.localizedDescription rangeOfString:expectedLocalizedDescription].location, \
      NSNotFound)

#define VerifyCategory(category, expectedIndex, expectedScore, expectedLabel, expectedDisplayName) \
  XCTAssertEqual(category.index, expectedIndex);                                                   \
  XCTAssertEqualWithAccuracy(category.score, expectedScore, 1e-6);                                 \
  XCTAssertEqualObjects(category.label, expectedLabel);                                            \
  XCTAssertEqualObjects(category.displayName, expectedDisplayName);


#define VerifyDetection(detection, expectedCategoryCount, expectedBoundingBox) \
  XCTAssertEqual(detection.categories.count, expectedCategoryCount);               \
  XCTAssertEqual(detection.boundingBox.origin.x, expectedBoundingBox.origin.x);                 \
  XCTAssertEqual(detection.boundingBox.origin.y, expectedBoundingBox.origin.y);                 \
  XCTAssertEqual(detection.boundingBox.size.width, expectedBoundingBox.size.width);             \
  XCTAssertEqual(detection.boundingBox.size.height, expectedBoundingBox.size.height)           
 

#define VerifyDetectionResult(detectionResult, expectedDetectionsCount) \
  XCTAssertNotNil(detectionResult);                                               \
  XCTAssertEqual(detectionResult.detections.count, expectedDetectionsCount)

static NSString *const expectedErrorDomain = @"org.tensorflow.lite.tasks";

@interface TFLObjectDetectorTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLObjectDetectorTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
  self.modelPath = [[NSBundle bundleForClass:[self class]]
      pathForResource:@"coco_ssd_mobilenet_v1_1.0_quant_2018_06_29"
               ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (void)testSuccessfullObjectDetectionOnMLImageWithUIImage {

  TFLObjectDetectorOptions *objectDetectorOptions =
      [[TFLObjectDetectorOptions alloc] initWithModelPath:self.modelPath];

  TFLObjectDetector *objectDetector =
      [TFLObjectDetector objectDetectorWithOptions:objectDetectorOptions error:nil];
  XCTAssertNotNil(objectDetector);

  GMLImage *gmlImage = [GMLImage imageFromBundleWithClass:self.class
                                                 fileName:@"cats_and_dogs"
                                                   ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLDetectionResult *detectionResult = [objectDetector detectWithGMLImage:gmlImage error:nil];

  const NSInteger expectedDetectionsCount = 10;
  NSLog(@"%d", detectionResult.detections.count );

  VerifyDetectionResult(detectionResult, expectedDetectionsCount);

  NSLog(@"%f", detectionResult.detections[0].categories[0].score );

  const NSInteger expectedCategoriesCount = 1;
  VerifyDetection(detectionResult.detections[0],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(54, 396, 393, 199)   // expectedBoundingBox
  );

  VerifyCategory(detectionResult.detections[0].categories[0],
                 16,       // expectedIndex
                 0.632812,  // expectedScore
                 @"cat",  // expectedLabel
                 nil        // expectedDisplaName
  );

  VerifyDetection(detectionResult.detections[1],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(602, 157, 394, 447)  // expectedBoundingBox
  );
  VerifyCategory(detectionResult.detections[1].categories[0],
                 16,       // expectedIndex
                 0.609375,  // expectedScore
                 @"cat",  // expectedLabel
                 nil        // expectedDisplaName
  );

  VerifyDetection(detectionResult.detections[2],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(260, 394, 179, 209)  // expectedBoundingBox
  );
  VerifyCategory(detectionResult.detections[2].categories[0],
                 16,       // expectedIndex
                 0.5625,  // expectedScore
                 @"cat",  // expectedLabel
                 nil        // expectedDisplaName
  );

  VerifyDetection(detectionResult.detections[3],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(387, 197, 281, 409)  // expectedBoundingBox
  );
  VerifyCategory(detectionResult.detections[3].categories[0],
                 17,       // expectedIndex
                 0.488281,  // expectedScore
                 @"dog",  // expectedLabel
                 nil        // expectedDisplaName
  );

}

- (void)testModelOptionsWithMaxResults {
   TFLObjectDetectorOptions *objectDetectorOptions =
      [[TFLObjectDetectorOptions alloc] initWithModelPath:self.modelPath];
  
  objectDetectorOptions.classificationOptions.maxResults = 4;

  TFLObjectDetector *objectDetector =
      [TFLObjectDetector objectDetectorWithOptions:objectDetectorOptions error:nil];
  XCTAssertNotNil(objectDetector);

  GMLImage *gmlImage = [GMLImage imageFromBundleWithClass:self.class
                                                 fileName:@"cats_and_dogs"
                                                   ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLDetectionResult *detectionResult = [objectDetector detectWithGMLImage:gmlImage error:nil];

  const NSInteger expectedDetectionsCount = 4;

  VerifyDetectionResult(detectionResult, expectedDetectionsCount);

  const NSInteger expectedCategoriesCount = 1;
  VerifyDetection(detectionResult.detections[0],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(54, 396, 393, 199)   // expectedBoundingBox
  );

  VerifyCategory(detectionResult.detections[0].categories[0],
                 16,       // expectedIndex
                 0.632812,  // expectedScore
                 @"cat",  // expectedLabel
                 nil        // expectedDisplaName
  );

  VerifyDetection(detectionResult.detections[1],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(602, 157, 394, 447)  // expectedBoundingBox
  );
  VerifyCategory(detectionResult.detections[1].categories[0],
                 16,       // expectedIndex
                 0.609375,  // expectedScore
                 @"cat",  // expectedLabel
                 nil        // expectedDisplaName
  );

  VerifyDetection(detectionResult.detections[2],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(260, 394, 179, 209)  // expectedBoundingBox
  );
  VerifyCategory(detectionResult.detections[2].categories[0],
                 16,       // expectedIndex
                 0.5625,  // expectedScore
                 @"cat",  // expectedLabel
                 nil        // expectedDisplaName
  );

  VerifyDetection(detectionResult.detections[3],
                  expectedCategoriesCount,        // expectedCategoriesCount
                  CGRectMake(387, 197, 281, 409)  // expectedBoundingBox
  );
  VerifyCategory(detectionResult.detections[3].categories[0],
                 17,       // expectedIndex
                 0.488281,  // expectedScore
                 @"dog",  // expectedLabel
                 nil        // expectedDisplaName
  );
}

- (void)testErrorForSimultaneousClassNameBlackListAndWhiteList {

  TFLObjectDetectorOptions *objectDetectorOptions =
      [[TFLObjectDetectorOptions alloc] initWithModelPath:self.modelPath];

  objectDetectorOptions.classificationOptions.labelDenyList =
      [NSArray arrayWithObjects:@"cat", nil];
  objectDetectorOptions.classificationOptions.labelAllowList =
      [NSArray arrayWithObjects:@"dog", nil];

  NSError *error = nil;
  
  TFLObjectDetector *objectDetector =
      [TFLObjectDetector objectDetectorWithOptions:objectDetectorOptions error:&error];
  XCTAssertNil(objectDetector);
  XCTAssertNotNil(error);

  const NSInteger expectedErrorCode = 2;
  NSString *const expectedLocalizedDescription =
      @"INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` are mutually exclusive "
      @"options";
  VerifyError(error,
              expectedErrorDomain,          // expectedDomain
              expectedErrorCode,            // expectedCode
              expectedLocalizedDescription  // expectedLocalizedDescription
  );
}

- (void)testErrorForInvalidMaxResults {

  TFLObjectDetectorOptions *objectDetectorOptions =
      [[TFLObjectDetectorOptions alloc] initWithModelPath:self.modelPath];
  objectDetectorOptions.classificationOptions.maxResults = 0;


  NSError *error = nil;
   TFLObjectDetector *objectDetector =
      [TFLObjectDetector objectDetectorWithOptions:objectDetectorOptions error:&error];
  XCTAssertNil(objectDetector);
  XCTAssertNotNil(error);

  const NSInteger expectedErrorCode = 2;
  NSString *const expectedLocalizedDescription =
      @"INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0";
  VerifyError(error,
              expectedErrorDomain,          // expectedDomain
              expectedErrorCode,            // expectedCode
              expectedLocalizedDescription  // expectedLocalizedDescription
  );
}

@end
