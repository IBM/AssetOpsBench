FROM python:3-slim

# install basic packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python-is-python3 \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN pip install virtualenv \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init

ARG CONDA_ENV_NAME=assetopsbench
ARG PYTHON_VERSION=3.12

RUN /opt/conda/bin/conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/main"
RUN /opt/conda/bin/conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/r"

# create conda environment and optionally install the requirements to it
RUN /opt/conda/bin/conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

COPY ./basic_requirements.txt /tmp/basic_requirements.txt

RUN /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install -r /tmp/basic_requirements.txt && \
    /opt/conda/bin/conda clean -afy

ARG INSTALL_EXTRA_DEPENDENCIES=true
ENV INSTALL_EXTRA_DEPENDENCIES=${INSTALL_EXTRA_DEPENDENCIES}

#ARG GITHUB_TOKEN
#ENV GITHUB_TOKEN=${GITHUB_TOKEN}

COPY AgentHive/ /tmp/AgentHive/
COPY FMSRAgent/ /tmp/FMSRAgent/
COPY IoTAgent/ /tmp/IoTAgent/
COPY ReActXen/ /tmp/ReActXen/
COPY TSFMAgent/ /tmp/TSFMAgent/
COPY WOAgent/ /tmp/WOAgent/
COPY MetaAgent/ /tmp/MetaAgent/

RUN /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install /tmp/ReActXen/
RUN cd /tmp/FMSRAgent/ && /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install .
RUN cd /tmp/IoTAgent/ && /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install .
RUN cd /tmp/WOAgent/ && /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install .

COPY granite-tsfm/ /tmp/granite-tsfm/
RUN opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install /tmp/granite-tsfm/

RUN cd /tmp/TSFMAgent/ && /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install .
RUN cd /tmp/AgentHive/ && /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install .
RUN cd /tmp/MetaAgent/ && /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install .

ENV PATH="/opt/conda/bin:${PATH}"

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
