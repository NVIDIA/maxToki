#!/bin/bash

for sub in ./3rdparty/*/; do
    pip install --break-system-packages --no-deps --no-build-isolation --editable $sub
done

for sub in ./sub-packages/bionemo-*/; do
    uv pip install --system --break-system-packages --link-mode=copy --no-deps --no-build-isolation --editable $sub
done
