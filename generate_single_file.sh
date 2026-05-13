#!/bin/bash
set -euo pipefail

cat ./*.hpp main.cpp > single_file/reproducer.cpp
./strip_local_header_includes.sh single_file/reproducer.cpp
