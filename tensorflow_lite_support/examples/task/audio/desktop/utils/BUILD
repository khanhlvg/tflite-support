package(
    default_visibility = [
        "//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "audio_utils",
    srcs = ["audio_utils.cc"],
    hdrs = ["audio_utils.h"],
    visibility = [
        "//tensorflow_lite_support:internal",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:integral_types",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/audio/utils:wav_io",
        "//tensorflow_lite_support/cc/task/audio/core:audio_buffer",
        "@com_google_absl//absl/status",
    ],
)
