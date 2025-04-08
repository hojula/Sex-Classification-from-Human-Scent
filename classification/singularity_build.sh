#!/bin/bash

PROJECT_DIR="/home/moq/some/scent_release/identification/"

singularity build "${PROJECT_DIR}/scent2.sif" "singularity_cu.def"