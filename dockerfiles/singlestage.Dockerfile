FROM python:3.10-slim-bookworm AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    ca-certificates \
    git \
    # nest-simulator dependencies, partly from https://nest-simulator.readthedocs.io/en/stable/installation/noenv_install.html#noenv
    libtool \
    cmake \
    gsl-bin \
    libgsl-dev \
    libltdl-dev \
    libboost-dev \
    libreadline-dev \
    libhdf5-dev \
    gcc \
    gosu \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sim
ENV VIRTUAL_ENV=/sim/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements_base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements_base.txt

ENV INSTALL_DIR=/sim/install
ENV DEPS_DIR=/sim/dependencies
ENV NEST_INSTALL_DIR=$INSTALL_DIR/nest
ENV MUSIC_INSTALL_DIR=$INSTALL_DIR/music

RUN mkdir -p $INSTALL_DIR $DEPS_DIR $NEST_INSTALL_DIR $MUSIC_INSTALL_DIR


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

# RUN pip install -r requirements_pynest.txt
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

ENV BULLET_MUSCLE_DIR=$DEPS_DIR/bullet_muscle_sim
ENV SDF_MODELS_DIR=/sim/embodiment_sdf_models
RUN mkdir -p $BULLET_MUSCLE_DIR $SDF_MODELS_DIR 


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
    pip install -r requirements.txt

RUN echo ${NEST_INSTALL_DIR}/lib/python3.10/site-packages >> ${VIRTUAL_ENV}/lib/python3.10/site-packages/nest.pth

# no default: this must be handled by builder
ARG USER_UID
ARG USER_GID
ENV USERNAME=simuser
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME

ENV CONTROLLER_DIR=/sim/controller
ENV SHARED_DATA_DIR=/sim/shared_data
ENV NEST_MODULE_PATH=/sim/install/nest/lib/nest/
RUN mkdir -p $CONTROLLER_DIR $SHARED_DATA_DIR $NEST_MODULE_PATH

RUN echo "Setting ownership of shared folders to: >>>>>>>>>>>>>>>>>>> userid:$USER_UID groupid:$USER_GID >>>>>>>>>>>>>>>>>>>"
RUN chown -R $USERNAME:$USERNAME $VIRTUAL_ENV /home/$USERNAME $SHARED_DATA_DIR $NEST_MODULE_PATH

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# COPY scripts/interactive_start.py /usr/local/bin/run_interactive.py
# RUN chmod +x /usr/local/bin/run_interactive.py

COPY scripts/aliases.sh /tmp/aliases.sh
RUN cat /tmp/aliases.sh >> /etc/bash.bashrc && rm /tmp/aliases.sh

# ENV LD_LIBRARY_PATH="$NEST_MODULE_PATH"
ENV PYTHONPATH="$CONTROLLER_DIR":"$BULLET_MUSCLE_DIR"
# music
ENV PATH="$PATH:$INSTALL_DIR/bin"

WORKDIR /sim


ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# CMD [ "python", "/usr/local/bin/run_interactive.py" ]