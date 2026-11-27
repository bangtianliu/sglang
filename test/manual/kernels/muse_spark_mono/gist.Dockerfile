# Separate CPU-built toolchain baseline; do not modify muse-mono-phase0.
FROM rocm/pytorch-private:rocm10.0_mi450mc_test_scxiao AS gist_stack
ARG TRITON_REV=b7ddcf86c095de79527ba3098503e73158bb94bd
ARG UTLX_REV=6c540caf36b022ff3772f6957db9b22416c709f7
ARG AITER_REV=9fe628bc646d9b014a4cdcc7feed4d3ada5270eb
ENV HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" \
    MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 TRITON_PARALLEL_LINK_JOBS=2 \
    TRITON_EXT_ENABLED=1 TRITON_BUILD_PROTON=OFF \
    LLVM_SYSPATH=/root/.triton/llvm/llvm-b010a18d-ubuntu-x64-1 \
    LLVM_INSTALL_DIR=/root/.triton/llvm/llvm-b010a18d-ubuntu-x64-1 \
    TRITON_SOURCE_DIR=/root/Workplace/projects/triton-openai \
    TRITON_WHEEL_DIR=/root/Workplace/projects/triton-openai/python/triton
RUN python3 -m pip install 'scikit-build-core>=0.11'
WORKDIR /root/Workplace/projects/triton-openai
RUN git fetch --depth=1 https://github.com/triton-lang/triton.git ${TRITON_REV} \
    && git checkout --detach FETCH_HEAD \
    && rm -rf build python/triton/include \
    && python3 -m pip install -v --no-build-isolation --no-deps -e .
RUN git clone --filter=blob:none --no-checkout https://github.com/triton-lang/triton-ext.git /opt/muse-gist/triton-ext \
    && git -C /opt/muse-gist/triton-ext checkout ${UTLX_REV} \
    && python3 -m pip install -v --no-build-isolation --no-deps /opt/muse-gist/triton-ext/extensions/utlx
WORKDIR /opt/muse-gist/aiter
RUN git init . \
    && git fetch --depth=1 https://github.com/ROCm/aiter.git ${AITER_REV} \
    && git checkout --detach FETCH_HEAD \
    && git clone --shared --no-checkout /root/Workplace/projects/aiter/3rdparty/composable_kernel 3rdparty/composable_kernel \
    && git -C 3rdparty/composable_kernel checkout 96afae1d54fc79c26857de7e1683b0dde1ff3c6f \
    && git -C 3rdparty/composable_kernel remote set-url origin https://github.com/ROCm/composable_kernel.git \
    && git submodule update --init --recursive --depth=1 \
    && AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 GPU_ARCHS=gfx1250 python3 -m pip install -v --no-build-isolation --no-deps -e .
RUN python3 -m pip install msgspec blake3 pybase64 requests tqdm aiohttp
FROM gist_stack AS gist_validated
ENV LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
COPY gist_sitecustomize.py /opt/venv/lib/python3.12/site-packages/muse_gist_loader.py
RUN printf "import muse_gist_loader\n" > /opt/venv/lib/python3.12/site-packages/00_muse_gist_loader.pth
RUN python3 -c "import torch,triton,utlx_plugin as t,importlib.metadata as m,subprocess; print('Triton',triton.__version__,m.version('triton'),triton.__file__,'uTLX',t.__file__); assert triton.__version__.startswith('3.9.0'); assert 'b7ddcf86' in m.version('triton'); assert subprocess.check_output(['git','-C','/root/Workplace/projects/triton-openai','rev-parse','HEAD'],text=True).strip()=='b7ddcf86c095de79527ba3098503e73158bb94bd'; assert hasattr(t,'tdm_async_load') and hasattr(t,'workgroup_barrier')"
LABEL muse.triton=b7ddcf86c095de79527ba3098503e73158bb94bd \
      muse.utlx=6c540caf36b022ff3772f6957db9b22416c709f7 \
      muse.aiter=9fe628bc646d9b014a4cdcc7feed4d3ada5270eb
WORKDIR /home/bangtliu/sglang-mono
ENV PYTHONPATH=/home/bangtliu/sglang-mono/python
