#!/usr/bin/env bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Set the following variables as appropriate.
#   * BAZEL: path to bazel. defaults to the first one available in PATH
#   * FRAMEWORK_NAME: name of the iOS framework to be built. Currently the
#   * accepted values are TensorFlowLiteTaskVision, TensorFlowLiteTaskText.
#   * TFLS_BUILD_VERSION: to specify the release version. defaults to 0.0.1-dev
#   * IS_RELEASE_BUILD: set as true if this build should be a release build
#   * ARCHIVE_FRAMEWORK: set as true if the framework should be archived
#   * DEST_DIR: destination directory to which the framework will be copied

set -ex

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This build script only works on macOS."
  exit 1
fi

BAZEL="${BAZEL:-$(which bazel)}"
TFLS_BUILD_VERSION=${TFLS_BUILD_VERSION:-0.0.1-dev}
TFLS_ROOT_DIR=$(git rev-parse --show-toplevel)
IOS_FRAMEWORK_PATH=""

if [[ ! -x "${BAZEL}" ]]; then
  echo "bazel executable is not found."
  exit 1
fi

if [ -z ${FRAMEWORK_NAME+x} ]; then
  echo "Name of the iOS framework, which is to be built, must be set."
  exit 1
fi

case $FRAMEWORK_NAME in
  "TensorFlowLiteTaskVision")
    ;;
  "TensorFlowLiteTaskText")
    ;;
  "TensorFlowLiteTaskAudio")
    ;;
  *)
    echo "Wrong framework name"
    exit 1
  ;;
esac

if [[ -z "${DEST_DIR+x}" || "${DEST_DIR}" == ${TFLS_ROOT_DIR}* ]]; then
  echo "DEST_DIR variable must be set and not be under the repository root."
  exit 1
fi

# get_output_file_path takes one bazel target label as an argument, and prints
# the path of the first output file of the specified target.
function get_output_file_path {
  local STARLARK_FILE="${OUTPUT_TMPDIR}/print_output_file.starlark"
  cat > "${STARLARK_FILE}" << EOF
def format(target):
  return target.files.to_list()[0].path
EOF
  bazel cquery --config=ios_fat $1 \
    --output=starlark --starlark:file="${STARLARK_FILE}" 2> /dev/null
}

function build_ios_api_framework {
  TARGET="//tensorflow_lite_support/ios"
  RULE_NAME="${TARGET}:$1_framework"

  "${BAZEL}" build -c opt --config=ios_fat ${RULE_NAME}
  
  OUTPUT_TMPDIR="$(mktemp -d)"

  IOS_FRAMEWORK_PATH="$(get_output_file_path "${RULE_NAME}")"

  # Clean up the temporary directory for the framework.
  rm -rf "${OUTPUT_TMPDIR}"
  echo ${OUTPUT_TMPDIR}
}

function create_framework_archive {
  TARGET_FRAMEWORK_NAME="$1"

  # Change to the Bazel iOS output directory.
  pushd "${BAZEL_IOS_OUTDIR}"

  # Create the temporary directory for the given framework.
  ARCHIVE_NAME="${TARGET_FRAMEWORK_NAME}-${TFLS_BUILD_VERSION}"
  TFLS_TMPDIR="$(mktemp -d)"

  # Copy the license file to TFLS_TMPDIR
  cp "LICENSE" ${TFLS_TMPDIR}
  unzip ${IOS_FRAMEWORK_PATH} -d "${TFLS_TMPDIR}"/Frameworks  

  # # FRAMEWORK_LOCATION=$(find "${TFLS_ROOT_DIR}/bazel-out/" -name ${TARGET_FRAMEWORK_NAME}_framework.zip)
  
  # # for path in ${FRAMEWORK_LOCATION}
  #   # do
  # unzip ${IOS_FRAMEWORK_PATH} -d "${TFLS_TMPDIR}"/Frameworks
  #     # break
  #   # done
  
  #----- (3) Move the framework to the destination -----
  if [[ "${ARCHIVE_FRAMEWORK}" == true ]]; then
    TARGET_DIR="$(realpath "${TARGET_FRAMEWORK_NAME}")"

    # Create the framework archive directory.
    if [[ "${IS_RELEASE_BUILD}" == true ]]; then
      # Get the first 16 bytes of the sha256 checksum of the root directory.
      SHA256_CHECKSUM=$(find "${TFLS_TMPDIR}" -type f -print0 | xargs -0 shasum -a 256 | sort | shasum -a 256 | cut -c1-16)
      FRAMEWORK_ARCHIVE_DIR="${TARGET_DIR}/${TFLS_BUILD_VERSION}/${SHA256_CHECKSUM}"
    else
      FRAMEWORK_ARCHIVE_DIR="${TARGET_DIR}/${TFLS_BUILD_VERSION}"
    fi
    mkdir -p "${FRAMEWORK_ARCHIVE_DIR}"

    # Zip up the framework and move to the archive directory.
    pushd "${TFLS_TMPDIR}"
    TFLS_ARCHIVE_FILE="${ARCHIVE_NAME}.tar.gz"
    tar -cvzf "${TFLS_ARCHIVE_FILE}" .
    mv "${TFLS_ARCHIVE_FILE}" "${FRAMEWORK_ARCHIVE_DIR}"
    popd

    # Move the target directory to the Kokoro artifacts directory.
    mv "${TARGET_DIR}" "$(realpath "${DEST_DIR}")"/
  else
    rsync -r "${TFLS_TMPDIR}/" "$(realpath "${DEST_DIR}")/"
  fi

  # Clean up the temporary directory for the framework.
  rm -rf "${TFLS_TMPDIR}"
  echo ${TFLS_TMPDIR}
}

cd "${TFLS_ROOT_DIR}"
build_ios_api_framework $FRAMEWORK_NAME
create_framework_archive $FRAMEWORK_NAME