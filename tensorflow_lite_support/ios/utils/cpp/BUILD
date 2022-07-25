package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLCommonCppUtils",
    srcs = [
        "sources/TFLCommonCppUtils.mm",
    ],
    hdrs = [
        "sources/TFLCommonCppUtils.h",
    ],
    module_name = "TFLCommonCppUtils",
    deps = [
        "//tensorflow_lite_support/cc:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "//tensorflow_lite_support/ios/utils:TFLCommonUtils",
    ],
)
