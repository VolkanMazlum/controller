# this unfortunately doesn't quite work yet. 

# root@a76e28484ef7:/sim# bsb
# Traceback (most recent call last):
#   File "/sim/venv/bin/bsb", line 8, in <module>
#     sys.exit(handle_cli())
#              ^^^^^^^^^^^^
#   File "/sim/venv/lib/python3.11/site-packages/bsb/cli/init.py", line 11, in handle_cli
#     handle_command(sys.argv[1:], exit=True)
#   File "/sim/venv/lib/python3.11/site-packages/bsb/cli/init.py", line 17, in handle_command
#     root_command = load_root_command()
#                    ^^^^^^^^^^^^^^^^^^^
#   File "/sim/venv/lib/python3.11/site-packages/bsb/cli/commands/init.py", line 150, in load_root_command
#     discover("commands")
#   File "/sim/venv/lib/python3.11/site-packages/bsb/plugins.py", line 56, in discover
#     errr.wrap(
#   File "/sim/venv/lib/python3.11/site-packages/errr/init.py", line 7, in wrap
#     return err_type.wrap(*args, kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/sim/venv/lib/python3.11/site-packages/errr/exception.py", line 132, in wrap
#     raise err from None
#   File "/sim/venv/lib/python3.11/site-packages/bsb/plugins.py", line 46, in discover
#     advert = entry.load()
#              ^^^^^^^^^^^^
#   File "/usr/local/lib/python3.11/importlib/metadata/init.py", line 202, in load
#     module = import_module(match.group('module'))
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.11/importlib/init.py", line 126, in import_module
#     return bootstrap.gcd_import(name[level:], package, level)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "<frozen importlib._bootstrap>", line 1204, in gcdimport
#   File "<frozen importlib._bootstrap>", line 1176, in findand_load
#   File "<frozen importlib._bootstrap>", line 1147, in findand_load_unlocked
#   File "<frozen importlib._bootstrap>", line 690, in loadunlocked
#   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
#   File "<frozen importlib._bootstrap>", line 241, in callwith_frames_removed
#   File "/sim/venv/lib/python3.11/site-packages/bsb/cli/commands/_commands.py", line 13, in <module>
#     from ...config import parse_configuration_file
#   File "/sim/venv/lib/python3.11/site-packages/bsb/config/init.py", line 41, in <module>
#     from ._distributions import Distribution
#   File "/sim/venv/lib/python3.11/site-packages/bsb/config/_distributions.py", line 5, in <module>
#     import scipy.stats.distributions as _distributions
#   File "/sim/venv/lib/python3.11/site-packages/scipy/stats/init.py", line 624, in <module>
#     from ._stats_py import *
#   File "/sim/venv/lib/python3.11/site-packages/scipy/stats/_stats_py.py", line 39, in <module>
#     from scipy.spatial import distance_matrix
#   File "/sim/venv/lib/python3.11/site-packages/scipy/spatial/init.py", line 116, in <module>
#     from ._geometric_slerp import geometric_slerp
#   File "/sim/venv/lib/python3.11/site-packages/scipy/spatial/_geometric_slerp.py", line 7, in <module>
#     from scipy.spatial.distance import euclidean
#   File "/sim/venv/lib/python3.11/site-packages/scipy/spatial/distance.py", line 121, in <module>
#     from ..special import rel_entr
#   File "/sim/venv/lib/python3.11/site-packages/scipy/special/init.py", line 826, in <module>
#     from . import _basic
#   File "/sim/venv/lib/python3.11/site-packages/scipy/special/_basic.py", line 22, in <module>
#     from ._multiufuncs import (assoc_legendre_p_all,
#   File "/sim/venv/lib/python3.11/site-packages/scipy/special/_multiufuncs.py", line 142, in <module>
#     sph_legendre_p = MultiUFunc(
#                      ^^^^^^^^^^^
#   File "/sim/venv/lib/python3.11/site-packages/scipy/special/_multiufuncs.py", line 41, in init**
#     raise ValueError("All ufuncs must have type numpy.ufunc."
# bsb.exceptions.PluginError: Could not instantiate the commands plugin:
# All ufuncs must have type numpy.ufunc. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)


# start from a common image:
# - python 3.11.11 (common python version for all components, we will create a venv)
# - slim version of Debian bookworm (pins the image to a specific version of Debian,
#   to avoid surprises from future updates; also, the slim version is smaller)

# the final image is built in multiple stages:
# - nest_builder: builds NEST and MUSIC
# - libs_builder: installs dependencies for the controller and bullet_muscle_sim
# - runner: the final image, with all dependencies installed

# note: NEST has non-python dependencies, and because i was unable to get the static
# build to work, they must be available in both nest_builder and runner stages.

# step 1: create common venv, install base requirements
# FROM python:3.11.11-slim-bookworm AS base_image
FROM python:3.10-slim-bookworm AS base_image

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sim
ENV VIRTUAL_ENV=/sim/venv
# note: i create venv here, but not all stages contribute to it
# this will be active in all following stages.
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements_base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_base.txt

# we'll put the binaries in $INSTALL_DIR, and the source code in $DEPS_DIR
ENV INSTALL_DIR=/sim/install
ENV DEPS_DIR=/sim/dependencies
ENV NEST_INSTALL_DIR=$INSTALL_DIR/nest
ENV MUSIC_INSTALL_DIR=$INSTALL_DIR/music

RUN mkdir -p $INSTALL_DIR $DEPS_DIR $NEST_INSTALL_DIR $MUSIC_INSTALL_DIR


# step 2: build NEST and MUSIC
FROM base_image AS nest_builder

# Install system build dependencies for MUSIC and NEST
RUN apt-get update && apt-get install -y --no-install-recommends \
    # general use
    build-essential \
    ca-certificates \
    git \
    # nest-simulator dependencies, partly from https://nest-simulator.readthedocs.io/en/stable/installation/noenv_install.html#noenv
    libtool \
    cmake \
    gsl-bin \
    libgsl-dev \
    libboost-dev \
    libreadline-dev \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

####################### MUSIC #######################
# Install MUSIC - pinned to last commit to ensure compatibility
WORKDIR $DEPS_DIR

RUN git clone https://github.com/INCF/MUSIC \
    && cd MUSIC \
    && git checkout 6d12edf7094e1dbb473dd55f6fcbd0825f30b811 \
    && ./autogen.sh \
    && ./configure --prefix=$INSTALL_DIR --with-python="$(which python)" \
    && make -j$(nproc) \
    && make install


####################### NEST 3.7 #######################
ENV NEST_VERSION=3.7

# Install NEST 3.7
WORKDIR $DEPS_DIR
RUN git clone --depth 1 --branch v${NEST_VERSION} https://github.com/nest/nest-simulator/
WORKDIR ${DEPS_DIR}/nest-simulator

RUN pip install -r requirements_pynest.txt
RUN mkdir -p nest-build \
    && cd nest-build \
    && cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST_INSTALL_DIR \
    -Dwith-mpi=ON \
    -Dwith-music=ON \
    -Dwith-openmp=ON \
    -Dwith-ltdl=ON \
    # -Dstatic-libraries=ON \ i tried this, but it didn't work. i don't think it's maintained
    # -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    $DEPS_DIR/nest-simulator \
    && make -j$(nproc) \
    && make install 

# step 3: build bullet_muscle_sim and install our dependencies.
# if you need to add another package, you'll add it in requirements.txt
FROM base_image AS libs_builder

ENV BULLET_MUSCLE_DIR=$DEPS_DIR/bullet_muscle_sim
ENV SDF_MODELS_DIR=/sim/embodiment_sdf_models
RUN mkdir -p $BULLET_MUSCLE_DIR $SDF_MODELS_DIR 

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install bullet muscle simulation - pinned... but not really useful
RUN git clone https://github.com/UM-Projects-MZ/bullet_muscle_sim.git $BULLET_MUSCLE_DIR \
    && cd $BULLET_MUSCLE_DIR \
    && git checkout f20f2564efc859774412c40a4554e3641cbf834c \
    && cd \
    && git clone https://github.com/UM-Projects-MZ/embodiment_sdf_models.git $SDF_MODELS_DIR \
    && cd $SDF_MODELS_DIR \
    && git checkout 226eba6989616c1505bbdf8c7dc8b93505aeeb0a

COPY requirements.txt . 
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# step 4: the final image which runs the simulation
FROM base_image AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    libltdl-dev \
    gsl-bin \
    libgsl-dev \
    libboost-dev \
    libreadline-dev \
    openmpi-bin \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV BULLET_MUSCLE_DIR=$DEPS_DIR/bullet_muscle_sim
ENV SDF_MODELS_DIR=/sim/embodiment_sdf_models
ENV CONTROLLER_DIR=/sim/controller
RUN mkdir -p $CONTROLLER_DIR

COPY --from=nest_builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=nest_builder $INSTALL_DIR $INSTALL_DIR
COPY --from=libs_builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=libs_builder $SDF_MODELS_DIR $SDF_MODELS_DIR
COPY --from=libs_builder $BULLET_MUSCLE_DIR $BULLET_MUSCLE_DIR

WORKDIR /sim

RUN echo ${NEST_INSTALL_DIR}/lib/python3.10/site-packages >> ${VIRTUAL_ENV}/lib/python3.10/site-packages/nest.pth

ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid $USER_GID simuser && \
    useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash simuser

RUN chown -R simuser:simuser /sim
USER simuser
WORKDIR /sim

# ENV LD_LIBRARY_PATH="$INSTALL_DIR/lib"
ENV PYTHONPATH=$PYTHONPATH:"$CONTROLLER_DIR":"$BULLET_MUSCLE_DIR"

COPY entrypoint.sh /sim/entrypoint.sh
RUN chmod +x /sim/entrypoint.sh
# TODO vertsion at line 157 should be consistent with base image
ENTRYPOINT [ "/sim/entrypoint.sh" ]