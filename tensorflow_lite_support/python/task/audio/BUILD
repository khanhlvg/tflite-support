# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "audio_classifier",
    srcs = [
        "audio_classifier.py",
    ],
    deps = [
        "//tensorflow_lite_support/python/task/core:task_options",
        "//tensorflow_lite_support/python/task/core:task_utils",
        "//tensorflow_lite_support/python/task/processor/proto:classification_options_pb2",
        "//tensorflow_lite_support/python/task/processor/proto:classifications_pb2",
        "//tensorflow_lite_support/python/task/audio/core:tensor_audio",
        "//tensorflow_lite_support/python/task/audio/core/pybinds:audio_utils",
        "//tensorflow_lite_support/python/task/audio/core/pybinds:audio_buffer",
        "//tensorflow_lite_support/python/task/audio/pybinds:_pywrap_audio_classifier",
        "//tensorflow_lite_support/python/task/audio/pybinds:audio_classifier_options_pb2",
    ],
)
