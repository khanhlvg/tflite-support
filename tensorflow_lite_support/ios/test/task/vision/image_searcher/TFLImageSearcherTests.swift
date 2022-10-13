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
import XCTest

@testable import TFLImageSearcher

class ImageSearcherTests: XCTestCase {

  static let bundle = Bundle(for: ImageSearcherTests.self)

  let kSearcherModelName = "mobilenet_v3_small_100_224_searcher"
  let kEmbedderModelName = "mobilenet_v3_small_100_224_embedder"
  let kMobileNetIndexName = "searcher_index"

  func validateError(
    _ error: Error,
    expectedLocalizedDescription: String
  ) {

    XCTAssertEqual(
      error.localizedDescription,
      expectedLocalizedDescription)
  }

  func validateNearestNeighbor(
    _ nearestNeighbor: NearestNeighbor,
    expectedMetadata: String,
    expectedDistance: Float
  ) {
    XCTAssertEqual(
      nearestNeighbor.metadata,
      expectedMetadata)
    XCTAssertEqual(
      nearestNeighbor.distance,
      expectedDistance,
      accuracy: 1e-6)
  }

  func validateSearchResultCount(
    _ searchResult: SearchResult,
    expectedNearestNeighborsCount: Int
  ) {
    XCTAssertEqual(
      searchResult.nearesteNeighbors.count,
      expectedNearestNeighborsCount)
  }

  func filePath(
    name name: String,
    extension: String,
  ) -> String? {
    
    let filePath = try XCTUnwrap(AudioClassifierTests.bundle.path(
        forResource: name,
        ofType: fileExtension))
    return filePath
  }

  func createImageSearcherOptions(
    modelName: String
  ) throws -> ImageSearcherOptions? {

    let modelPath = filePath(name:modelName
                             extension:"tflite")
    return ImageSearcherOptions(modelPath: modelPath)
  }

  func createImageSearcher(
    _ modelName: String
    indexFileName: String? = nil
  ) throws -> ImageSearcher? {
    let options = try XCTUnwrap(
      self.createImageSearcherOptions(
        modelPath: ImageSearcherTests.modelPath))
    

    if let _indexFileName = indexFileName {
      let indexFilePath = filePath(name:indexFileName
                             extension:"ldb")
      options.searchOptions.indexFile.filePath = indexFilePath
    }

    let imageSearcher = try XCTUnwrap(
      ImageSearcher.searcher(
        options: options))

    return imageSearcher
  }
 

  func validateSearchResult(
    searchResult: SearchResult
  ) {
    self.validateSearchResultCount(
      searchResult,
      expectedNearestNeighborsCount: 5)
    
    self.verifyNearestNeighbor(
      searchResult.nearesteNeighbors[0],
      expectedMetadata: "burger",
      expectedDistance: 198.456329)
    self.verifyNearestNeighbor(
      searchResult.nearesteNeighbors[1],
      expectedMetadata: "car",
      expectedDistance: 226.022186)
    
    self.verifyNearestNeighbor(
      searchResult.nearestNeighbors[2],
      expectedMetadata: "bird",
      expectedDistance: 227.297668)
    self.verifyNearestNeighbor(
      searchResult.nearestNeighbors[3],
      expectedMetadata: "dog",
      expectedDistance: 229.133789)
   self.verifyNearestNeighbor(searchResult.nearestNeighbors[4],
      expectedMetadata: "cat", 
      expectedDistance: 229.718948)
  }

  func testSearchWithSearcherModelSucceeds() throws {
    let imageSearcher = try XCTUnwrap(
      self.createImageSearcher(
        modelName: kSearcherModelName)
   
    let mlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let searchResult = try XCTUnwrap(
      imageSearcher.search(
        mlImage: mlImage)
    self.validateSearchResult(searchResult)
  }

  func testCreateImageSearcherWithNoModelPathFails() throws {
    let options = ImageSearcherOptions()
    do {
      let imageSearcher = try ImageSearcher.searcher(
        options: options)
      XCTAssertNil(imageSearcher)
    } catch {
      self.verifyError(
        error,
        expectedLocalizedDescription:
          "INVALID_ARGUMENT: Missing mandatory `model_file` field in `base_options`")
    }
  }
}
