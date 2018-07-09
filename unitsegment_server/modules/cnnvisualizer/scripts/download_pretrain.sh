#!/usr/bin/env sh

SNAPSHOTS_BASE='../snapshots'
WIDERESNET18_BASE="$SNAPSHOTS_BASE/wideresnet18"

if [ ! -d ${SNAPSHOTS_BASE} ]; then
    mkdir ${SNAPSHOTS_BASE}
fi

wget http://places2.csail.mit.edu/models_places365/whole_wideresnet18_places365.pth.tar -P ${WIDERESNET18_BASE}
wget https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt -P ${WIDERESNET18_BASE} 
