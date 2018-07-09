#!/usr/bin/env sh

DATASETS_BASE='../datasets'
IMAGES_PATH="${DATASETS_BASE}/images.tar"

if [ ! -d ${DATASETS_BASE} ]; then
    mkdir ${DATASETS_BASE}
fi

if [ ! -f "${DATASETS_BASE}/images.tar" ]; then
    wget http://places.csail.mit.edu/unit_annotation/data/images.tar -O ${IMAGES_PATH}
fi

tar -xvf ${IMAGES_PATH} -C ${DATASETS_BASE}

