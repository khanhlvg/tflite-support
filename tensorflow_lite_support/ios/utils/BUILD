package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLStringUtil",
    srcs = [
        "Sources/TFLStringUtil.mm",
    ],
    hdrs = [
        "Sources/TFLStringUtil.h",
    ],
    module_name = "TFLStringUtil",
)

objc_library(
    name = "TFLCommonUtils",
    srcs = [
        "sources/TFLCommonUtils.mm",
    ],
    hdrs = [
        "sources/TFLCommonUtils.h",
    ],
    module_name = "TFLCommonUtils",
    deps = [
        "//tensorflow_lite_support/c:common",
        "//tensorflow_lite_support/ios:TFLCommon",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)
