#!/bin/bash

for f in ./sourcePdfs/*.pdf; do
    echo "Processing $f"
    python store.py "$f"
done
