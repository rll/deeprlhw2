#!/bin/bash
set -e
cd vendor/Arcade-Learning-Environment && mkdir -p build && cd build && cmake .. && make -j4
