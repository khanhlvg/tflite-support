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
import GMLImageUtils
import XCTest

@testable import TFLObjectDetector

class TFLObjectDetectorTests: XCTestCase {

  static let bundle = Bundle(for: TFLObjectDetectorTests.self)
  static let modelPath = bundle.path(
    forResource: "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29",
    ofType: "tflite")!

  func verifyError(
    _ error: Error,
    expectedLocalizedDescription: String
  ) {
    XCTAssert(
      error.localizedDescription.contains(expectedLocalizedDescription))
  }

  func verifyCategory(
    _ category: TFLCategory,
    expectedIndex: NSInteger,
    expectedScore: Float,
    expectedLabel: String,
    expectedDisplayName: String?
  ) {
    XCTAssertEqual(
      category.index,
      expectedIndex)
    XCTAssertEqual(
      category.score, 
      expectedScore, 
      accuracy: 1e-6); 

    XCTAssertEqual(
      category.label,
      expectedLabel)
    
    XCTAssertEqual(
      category.displayName,
      expectedDisplayName)
  }

  func verifyDetection(
    _ detection: TFLDetection,
    expectedCategoryCount: NSInteger,
    expectedBoundingBox: CGRect
  ) {
    XCTAssertEqual(
      detection.categories.count,
      expectedCategoryCount)
    
    XCTAssertEqual(
      detection.boundingBox.origin.x,
      expectedBoundingBox.origin.x)
    XCTAssertEqual(
      detection.boundingBox.origin.y,
      expectedBoundingBox.origin.y)
    XCTAssertEqual(
      detection.boundingBox.size.width,
      expectedBoundingBox.size.width)
    XCTAssertEqual(
      detection.boundingBox.size.height,
      expectedBoundingBox.size.height)
  }

  func verifyDetectionResult(
    _ detectionResult: TFLDetectionResult,
    expectedDetectionsCount: NSInteger
  ) {
    XCTAssertEqual(
      detectionResult.detections.count,
      expectedDetectionsCount)
  }


  func testObjectDetectionOnMLImageWithUIImage() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "cats_and_dogs",
        type: "jpg"))
    let detectionResult: TFLDetectionResult =
      try objectDetector.detect(gmlImage: gmlImage)

    let expectedDetectionsCount = 10
    self.verifyDetectionResult(detectionResult, 
                                    expectedDetectionsCount: expectedDetectionsCount)

    let expectedCategoryCount = 1
    self.verifyDetection(detectionResult.detections[0], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 54, y: 396, width: 393, height: 199) )
  
    // TODO: match the score as image_classifier_test.cc
    self.verifyCategory(detectionResult.detections[0].categories[0], 
                        expectedIndex: 16, 
                        expectedScore: 0.632812,
                        expectedLabel: "cat", 
                        expectedDisplayName: nil);

    self.verifyDetection(detectionResult.detections[1], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 602, y: 157, width: 394, height: 447))
    self.verifyCategory(detectionResult.detections[1].categories[0], 
                        expectedIndex: 16, 
                        expectedScore: 0.609375,
                        expectedLabel: "cat", 
                        expectedDisplayName: nil);

    
    self.verifyDetection(detectionResult.detections[2], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 260, y: 394, width: 179, height: 209))
    self.verifyCategory(detectionResult.detections[2].categories[0], 
                        expectedIndex: 16, 
                        expectedScore: 0.5625,
                        expectedLabel: "cat", 
                        expectedDisplayName: nil);                       
    
    self.verifyDetection(detectionResult.detections[3], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 387, y: 197, width: 281, height: 409))
    self.verifyCategory(detectionResult.detections[3].categories[0], 
                        expectedIndex: 17, 
                        expectedScore: 0.488281,
                        expectedLabel: "dog", 
                        expectedDisplayName: nil);  
  }

  func testModelOptionsWithMaxResults() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    objectDetectorOptions!.classificationOptions.maxResults = 4

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "cats_and_dogs",
        type: "jpg"))
    let detectionResult: TFLDetectionResult = try objectDetector.detect(
      gmlImage: gmlImage)

    self.verifyDetectionResult(detectionResult, 
                                    expectedDetectionsCount: objectDetectorOptions!.classificationOptions.maxResults)

    let expectedCategoryCount = 1
    self.verifyDetection(detectionResult.detections[0], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 54, y: 396, width: 393, height: 199) )
  
    // TODO: match the score as image_classifier_test.cc
    self.verifyCategory(detectionResult.detections[0].categories[0], 
                        expectedIndex: 16, 
                        expectedScore: 0.632812,
                        expectedLabel: "cat", 
                        expectedDisplayName: nil);

    self.verifyDetection(detectionResult.detections[1], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 602, y: 157, width: 394, height: 447))
    self.verifyCategory(detectionResult.detections[1].categories[0], 
                        expectedIndex: 16, 
                        expectedScore: 0.609375,
                        expectedLabel: "cat", 
                        expectedDisplayName: nil);

    
    self.verifyDetection(detectionResult.detections[2], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 260, y: 394, width: 179, height: 209))
    self.verifyCategory(detectionResult.detections[2].categories[0], 
                        expectedIndex: 16, 
                        expectedScore: 0.5625,
                        expectedLabel: "cat", 
                        expectedDisplayName: nil);                       
    
    self.verifyDetection(detectionResult.detections[3], 
                               expectedCategoryCount: expectedCategoryCount,
                               expectedBoundingBox: CGRect(x: 387, y: 197, width: 281, height: 409))
    self.verifyCategory(detectionResult.detections[3].categories[0], 
                        expectedIndex: 17, 
                        expectedScore: 0.488281,
                        expectedLabel: "dog", 
                        expectedDisplayName: nil);  
  }

  func testErrorForSimultaneousLabelAllowListAndDenyList() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = try XCTUnwrap(TFLObjectDetectorOptions(modelPath: modelPath))
    objectDetectorOptions.classificationOptions.labelAllowList = ["cheeseburger"];
    objectDetectorOptions.classificationOptions.labelDenyList = ["cheeseburger"];

    do {
      let objectDetector =
        try TFLObjectDetector.objectDetector(options: objectDetectorOptions)
      XCTAssertNil(objectDetector)
    }
    catch  {
      let expectedLocalizedDescription =
        "INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` are mutually exclusive options"
      self.verifyError(error,
                       expectedLocalizedDescription: expectedLocalizedDescription)
    }
  }

  func testErrorForOptionsWithInvalidMaxResults() throws { 
    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = try XCTUnwrap(TFLObjectDetectorOptions(modelPath: modelPath))
    
    let maxResults = 0
    objectDetectorOptions.classificationOptions.maxResults = maxResults
    
    do {
      let objectDetector =
        try TFLObjectDetector.objectDetector(options: objectDetectorOptions)
      XCTAssertNil(objectDetector)
    }
    catch {
      let expectedLocalizedDescription =
        "INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0"
      self.verifyError(error,
                       expectedLocalizedDescription: expectedLocalizedDescription)
    }
  }

    
}
