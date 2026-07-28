#!/bin/bash

# Only Megatron-LM is expected in 3rdparty (see Dockerfile). Globbing ./3rdparty/*/ would pick up
# a stray NeMo source clone and shadow the PyPI nemo_toolkit, breaking imports.
for sub in ./3rdparty/Megatron-LM/; do
    pip install --break-system-packages --no-deps --no-build-isolation --editable $sub
done

for sub in ./sub-packages/bionemo-*/; do
    uv pip install --system --break-system-packages --link-mode=copy --no-deps --no-build-isolation --editable $sub
done
