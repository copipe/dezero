ARG BASE_IMAGE=nvidia/cuda:11.0.3-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=ascender
ARG USER_NAME=challenger
ARG GROUP_NAME=challengers
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.8
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

# Following is needed to install python 3.7
RUN apt update && apt install --no-install-recommends -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt update && apt install --no-install-recommends -y \
    git curl make ssh openssh-client vim tmux graphviz\
    python${PYTHON_VERSION} python3-pip python-is-python3

# Following is needed to swtich default python3 version
# For detail, please check following link https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    # `requests` needs to be upgraded to avoid RequestsDependencyWarning
    # ref: https://stackoverflow.com/questions/56155627/requestsdependencywarning-urllib3-1-25-2-or-chardet-3-0-4-doesnt-match-a-s
    && python3 -m pip install --upgrade pip setuptools requests \
    && python3 -m pip install --pre poetry

# Add user. Without this, following process is executed as admin. 
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}
USER ${USER_NAME}

WORKDIR ${APPLICATION_DIRECTORY}