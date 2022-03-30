Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteTaskVision'
  s.version          = '0.0.1'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tflite-support'
  s.source           = { :http => "file:///Users/priankakariat/Desktop/tfl-ios-2/TensorFlowLiteTaskVision-0.0.1-dev.tar.gz"}
  s.summary          = 'TensorFlow Lite Task Library - Vision'
  s.description      = 'The Natural Language APIs of the TFLite Task Library'

  s.ios.deployment_target = '13.0'

  s.module_name = 'TensorFlowLiteTaskVision'
  s.static_framework = true

  objc_dir =  'tensorflow_lite_support/ios/'
  objc_task_dir =  objc_dir + 'task/'
  objc_core_dir = objc_task_dir + 'core/'
  objc_processor_dir = objc_task_dir + 'processor/'
  objc_vision_dir = objc_task_dir + 'vision/'
  gml_image_dir = 'tensorflow_lite_support/odml/ios/image/'
  s.public_header_files = [
    objc_vision_dir + 'apis/*.h',
    objc_dir + 'sources/TFLCommon.h',
    objc_core_dir + 'sources/TFLBaseOptions.h',
    objc_processor_dir + 'sources/{TFLSegmentationResult}.h',
    objc_vision_dir + 'sources/*.h',
    gml_image_dir + 'apis/*.h'
  ]

  c_dir = 'tensorflow_lite_support/c/'
  c_task_dir = 'tensorflow_lite_support/c/task/'
  c_core_dir = c_task_dir + 'core/'
  c_processor_dir = c_task_dir + 'processor/'
  c_vision_dir = c_task_dir + 'vision/'
  s.source_files = [
    c_dir + '*.h',
    c_core_dir + '*.h',
    c_processor_dir + '{category,bounding_box,classification_options,classification_result,detection_result,segmentation_result}.h',
    c_vision_dir + '*.h',
    c_vision_dir  + 'core/*.h',
    objc_dir + 'sources/*',
    objc_core_dir + 'sources/*',
    objc_processor_dir + 'sources/*',
    objc_vision_dir + 'apis/',
    objc_vision_dir + '*/sources/*',
    objc_vision_dir + 'sources/*',
    gml_image_dir + 'apis/*.h',
    gml_image_dir + 'sources/*.h'
  ]

  s.module_map = objc_vision_dir + 'apis/framework.modulemap'
  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' =>
      '"${PODS_TARGET_SRCROOT}" ' +
      '"${PODS_TARGET_SRCROOT}/' + c_dir + '" ' +
      '"${PODS_TARGET_SRCROOT}/' + c_dir + 'core" ' +
      '"${PODS_TARGET_SRCROOT}/' + c_dir + 'processor" ' +
      '"${PODS_TARGET_SRCROOT}/' + c_dir + 'vision" ' +
      '"${PODS_TARGET_SRCROOT}/' + c_dir + 'vision/core" ' +
      '"${PODS_TARGET_SRCROOT}/' + gml_image_dir + 'apis" ' +
      '"${PODS_TARGET_SRCROOT}/' + objc_dir + 'sources" ' +
      '"${PODS_TARGET_SRCROOT}/' + objc_core_dir + 'sources" ' +
      '"${PODS_TARGET_SRCROOT}/' + objc_processor_dir + 'sources" ' +
      '"${PODS_TARGET_SRCROOT}/' + objc_vision_dir + 'apis" ' +
      '"${PODS_TARGET_SRCROOT}/' + objc_vision_dir + 'sources" ',
    'VALID_ARCHS' => 'x86_64 armv7 arm64',
  }

  s.library = 'c++'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteTaskVisionC.framework'
end
