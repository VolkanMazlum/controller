ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=10

FROM python:${PYTHON_MAJOR}.${PYTHON_MINOR}-slim-bookworm AS runner

ARG PYTHON_MAJOR
ARG PYTHON_MINOR

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
    # --- VNC Dependencies ---
    tigervnc-standalone-server \
    tigervnc-common \
    tigervnc-tools \
    fluxbox \
    xterm \
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
    && ./configure --prefix=$MUSIC_INSTALL_DIR --with-python=$VIRTUAL_ENV/bin/python \
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
    -Dwith-music=$MUSIC_INSTALL_DIR \
    -Dwith-python=ON \
    # just to make extra extra sure
    -DPYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python \ 
    -Dwith-openmp=ON \
    -Dwith-ltdl=ON \
    # -Dstatic-libraries=ON \ i tried this, but it didn't work. i don't think it's maintained
    # -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    $DEPS_DIR/nest-simulator \
    && make -j$(nproc) \
    && make install 

ENV PYNEST_QUIET=1

ENV BULLET_MUSCLE_DIR=$DEPS_DIR/bullet_muscle_sim
ENV SDF_MODELS_DIR=/sim/embodiment_sdf_models
RUN mkdir -p $BULLET_MUSCLE_DIR $SDF_MODELS_DIR 


# Install bullet muscle simulation
RUN git clone https://github.com/near-nes/bullet_muscle_sim.git $BULLET_MUSCLE_DIR \
    && git clone https://github.com/near-nes/embodiment_sdf_models.git $SDF_MODELS_DIR

WORKDIR ${DEPS_DIR}
COPY requirements.txt . 
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

ENV PYTHON_MAJOR_MINOR="python${PYTHON_MAJOR}.${PYTHON_MINOR}"

RUN echo ${NEST_INSTALL_DIR}/lib/${PYTHON_MAJOR_MINOR}/site-packages >> ${VIRTUAL_ENV}/lib/${PYTHON_MAJOR_MINOR}/site-packages/nest.pth

# no default: this must be handled by builder
ARG USER_UID
ARG USER_GID
ENV USERNAME=simuser
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME

ENV CONTROLLER_DIR=/sim/controller
ENV SHARED_DATA_DIR=/sim/shared_data
ENV NEST_MODULE_PATH=/sim/install/nest/lib/nest
RUN mkdir -p $CONTROLLER_DIR $SHARED_DATA_DIR $NEST_MODULE_PATH

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY scripts/prepare_python_env.sh /usr/local/bin/prepare_python_env.sh
RUN chmod +x /usr/local/bin/prepare_python_env.sh

# >>>>>>>>>> VNC SETUP >>>>>>>>>>>>>>
ENV VNC_DISPLAY=":1"
ENV DISPLAY="${VNC_DISPLAY}"
COPY scripts/start-vnc.sh /usr/local/bin/start-vnc.sh
RUN chmod +x /usr/local/bin/start-vnc.sh

RUN mkdir -p /home/simuser/.vnc
COPY scripts/xstartup /home/simuser/.vnc/xstartup
RUN chmod 700 /home/$USERNAME/.vnc/xstartup

# COPY scripts/interactive_start.py /usr/local/bin/run_interactive.py
# RUN chmod +x /usr/local/bin/run_interactive.py

COPY scripts/aliases.sh /tmp/aliases.sh
RUN cat /tmp/aliases.sh >> /etc/bash.bashrc && rm /tmp/aliases.sh
# Set display for terminals opened in VNC
RUN echo "export DISPLAY=${VNC_DISPLAY}" >> /etc/bash.bashrc 

ENV CEREBELLUM_PATH="${CONTROLLER_DIR}/cerebellum"
ENV LD_LIBRARY_PATH="${MUSIC_INSTALL_DIR}/lib"
ENV PATH="${VIRTUAL_ENV}/bin:${MUSIC_INSTALL_DIR}/bin:${NEST_INSTALL_DIR}/bin:${PATH}"

ENV PYTHONPATH="${CONTROLLER_DIR}:${BULLET_MUSCLE_DIR}:${CEREBELLUM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
RUN echo "Constructed PYTHONPATH: ${PYTHONPATH}"

WORKDIR /sim

RUN echo "Setting ownership of shared folders to: >>>>>>>>>>>>>>>>>>> userid:$USER_UID groupid:$USER_GID >>>>>>>>>>>>>>>>>>>"
RUN chown -R $USERNAME:$USERNAME $VIRTUAL_ENV /home/$USERNAME $SHARED_DATA_DIR $NEST_MODULE_PATH $BULLET_MUSCLE_DIR


# Expose VNC port (Display :1 is 5901)
EXPOSE 5901
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# CMD [ "python", "/usr/local/bin/run_interactive.py" ]