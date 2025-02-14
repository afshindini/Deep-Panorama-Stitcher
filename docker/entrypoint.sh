#!/bin/bash -l
set -e
if [ "$#" -eq 0 ]; then
  exec panaroma_stitcher -vv -d ./test_data/boat opencv-simple --stitcher_type panorama
else
  exec "$@"
fi
