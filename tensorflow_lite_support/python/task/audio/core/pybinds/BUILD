load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "audio_utils",
    srcs = [
        "audio_utils.cc",
    ],
    module_name = "audio_utils",
    deps = [
        "//tensorflow_lite_support/examples/task/audio/desktop/utils:audio_utils",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

pybind_extension(
    name = "audio_buffer",
    srcs = [
        "audio_buffer.cc",
    ],
    module_name = "audio_buffer",
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/audio/core:audio_buffer",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
