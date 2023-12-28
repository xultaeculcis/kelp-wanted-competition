FROM ubuntu:22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Setup environment to match variables set by repo2docker as much as possible
# The name of the conda environment into which the requested packages are installed
ENV CONDA_ENV=kelp \
    # Tell apt-get to not block installs by asking for interactive human input
    DEBIAN_FRONTEND=noninteractive \
    # Set username, uid and gid (same as uid) of non-root user the container will be run as
    APP_USER=jovyan \
    APP_UID=1000 \
    # Use /bin/bash as shell, not the default /bin/sh (arrow keys, etc don't work then)
    SHELL=/bin/bash \
    # Setup locale to be UTF-8, avoiding gnarly hard to debug encoding errors
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    # Install conda in the same place repo2docker does
    CONDA_DIR=/srv/conda

# All env vars that reference other env vars need to be in their own ENV block
# Path to the python environment where the packages are installed
ENV ENV_PYTHON_PREFIX=${CONDA_DIR}/envs/${CONDA_ENV} \
    # Home directory of our non-root user
    HOME=/home/${APP_USER}

# Add both our Kelp env as well as default conda installation to $PATH
# Thus, when we start a `python` process, it loads the python in the ps conda environment,
# as that comes first here.
ENV PATH=${ENV_PYTHON_PREFIX}/bin:${CONDA_DIR}/bin:${PATH}

RUN echo "Creating ${APP_USER} user..." \
    # Create a group for the user to be part of, with gid same as uid
    && groupadd --gid ${APP_UID} ${APP_USER}  \
    # Create non-root user, with given gid, uid and create $HOME
    && useradd --create-home --gid ${APP_UID} --no-log-init --uid ${APP_UID} ${APP_USER} \
    # Make sure that /srv is owned by non-root user, so we can install things there
    && chown -R ${APP_USER}:${APP_USER} /srv

# Run conda activate each time a bash shell starts, so users don't have to manually type conda activate
# Note this is only read by shell, but not by the jupyter notebook - that relies
# on us starting the correct `python` process, which we do by adding the notebook conda environment's
# bin to PATH earlier ($ENV_PYTHON_PREFIX/bin)
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh ; conda activate ${CONDA_ENV}" > /etc/profile.d/init_conda.sh

# Install basic apt packages
RUN echo "Installing Apt-get packages..." \
    && apt-get update --fix-missing > /dev/null \
    && apt-get install -y apt-utils wget zip tzdata > /dev/null \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add TZ configuration - https://github.com/PrefectHQ/prefect/issues/3061
ENV TZ UTC
# ========================

USER ${APP_USER}
WORKDIR ${HOME}

# Install latest mambaforge in ${CONDA_DIR}
RUN echo "Installing Mambaforge..." \
    && URL="https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" \
    && wget --quiet ${URL} -O installer.sh \
    && /bin/bash installer.sh -u -b -p ${CONDA_DIR} \
    && rm installer.sh \
    && mamba install conda-lock -y \
    && mamba clean -afy \
    # After installing the packages, we cleanup some unnecessary files
    # to try reduce image size - see https://jcristharif.com/conda-docker-tips.html
    # Although we explicitly do *not* delete .pyc files, as that seems to slow down startup
    # quite a bit unfortunately - see https://github.com/2i2c-org/infrastructure/issues/2047
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete

COPY --chown=jovyan:jovyan . /home/jovyan

# We want to keep our images as reproducible as possible. If a lock
# file with exact versions of all required packages is present, we use
# it to install packages. conda-lock (https://github.com/conda-incubator/conda-lock)
# file - so we get the exact same versions each time the image is built. This
# also lets us see what packages have changed between two images by diffing
# the contents of the lock file between those image versions.
# After installing the packages, we cleanup some unnecessary files
# to try reduce image size - see https://jcristharif.com/conda-docker-tips.html
RUN conda-lock install --name ${CONDA_ENV} conda-lock.yml && \
    mamba clean -yaf && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete && \
    if [ -d ${ENV_PYTHON_PREFIX}/lib/python*/site-packages/bokeh/server/static ]; then \
    find ${ENV_PYTHON_PREFIX}/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete \
    ; fi

CMD ["bash"]
