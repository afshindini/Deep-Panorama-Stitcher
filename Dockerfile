## syntax=docker/dockerfile:1.1.7-experimental

################
# Base builder #
################
FROM python:3.10-bookworm as base_build

ENV \
  # locale environment variables
  LC_ALL=C.UTF-8 \
  # python environemnt variables
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  # pip environmental variables
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  # poetry version
  POETRY_VERSION=1.5.0

# Install requirements
RUN apt-get update && apt-get install -y \
        curl \
        git \
        bash \
        build-essential \
        libffi-dev \
        libssl-dev \
        tini \
        openssh-client \
        cargo \
        musl-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    # github ssh key setting
    && mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com | sort > ~/.ssh/known_hosts \
    # Installing poetry and set the PATH
    && curl -sSL https://install.python-poetry.org | python3 - \
    && echo 'export PATH="/root/.local/bin:$PATH"' >>/root/.profile \
    && export PATH="/root/.local/bin:$PATH" \
    && true
SHELL ["/bin/bash", "-lc"]

# Copy poetry lock and pyproject config files to the container
WORKDIR /pysetup
COPY ./poetry.lock ./pyproject.toml /pysetup/
# Install pip/wheel/virtualenv and build the wheels based on the poetry lock
RUN --mount=type=ssh pip3 install wheel virtualenv poetry-plugin-export \
    && poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt \
    && pip3 wheel --wheel-dir=/tmp/wheelhouse --trusted-host 172.17.0.1 --find-links=http://172.17.0.1:3141/debian/ -r /tmp/requirements.txt \
    && virtualenv /.venv && source /.venv/bin/activate && echo 'source /.venv/bin/activate' >>/root/.profile \
    && pip3 install --no-deps --trusted-host 172.17.0.1 --find-links=http://172.17.0.1:3141/debian/ --find-links=/tmp/wheelhouse/ /tmp/wheelhouse/*.whl \
    && true


###########################
# Production base builder #
###########################
FROM base_build as production_build
# Copy entrypoint script to the container and src files to the app directory
COPY ./docker/entrypoint.sh /docker-entrypoint.sh
COPY . /app/
WORKDIR /app
# Build the wheel packages with poetry and add them to the wheelhouse
RUN --mount=type=ssh source /.venv/bin/activate \
    && poetry build -f wheel --no-interaction --no-ansi \
    && cp dist/*.whl /tmp/wheelhouse \
    && chmod a+x /docker-entrypoint.sh \
    && true



########################
# Production Container #
########################
FROM python:3.10-bookworm as production
COPY --from=production_build /tmp/wheelhouse /tmp/wheelhouse
COPY --from=production_build /docker-entrypoint.sh /docker-entrypoint.sh
WORKDIR /app
# Install system level deps for running the package and install the wheels we built in the previous step.
RUN --mount=type=ssh apt-get update && apt-get install -y \
        bash \
        libffi8 \
        libgl1 \
        tini \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && chmod a+x /docker-entrypoint.sh \
    && WHEELFILE=`echo /tmp/wheelhouse/panaroma_stitcher-*.whl` \
    && pip3 install --trusted-host 172.17.0.1 --find-links=http://172.17.0.1:3141/debian/ --find-links=/tmp/wheelhouse/ "$WHEELFILE"[all] \
    && rm -rf /tmp/wheelhouse/ \
    && true
ENTRYPOINT ["/usr/bin/tini", "--", "/docker-entrypoint.sh"]



############################
# Development base builder #
############################
FROM base_build as development_build
# Copy src to app directory
COPY . /app
WORKDIR /app
# Install dependencies from poetry lock
RUN --mount=type=ssh source /.venv/bin/activate \
    && apt-get update && apt-get install -y libgl1 \
    && export PIP_FIND_LINKS=http://172.17.0.1:3141/debian/ \
    && export PIP_TRUSTED_HOST=172.17.0.1 \
    && pip3 install nvidia-cublas-cu12 nvidia-cusparse-cu12 triton nvidia-nccl-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 \
    && poetry install --no-interaction --no-ansi \
    && true



###################
# Tests Container #
###################
FROM development_build as test
RUN --mount=type=ssh source /.venv/bin/activate \
    && chmod a+x docker/*.sh \
    && docker/pre_commit_init.sh \
    && true
ENTRYPOINT ["/usr/bin/tini", "--", "docker/entrypoint-test.sh"]


#########################
# Development Container #
#########################
FROM development_build as development
RUN apt-get update && apt-get install -y zsh \
    && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && echo "if [ \"\$NO_WHEELHOUSE\" = \"1\" ]" >>/root/.profile \
    && echo "then" >>/root/.profile \
    && echo "  echo \"Wheelhouse disabled\"" >>/root/.profile \
    && echo "else">>/root/.profile \
    && echo "  export PIP_TRUSTED_HOST=172.17.0.1" >>/root/.profile \
    && echo "  export PIP_FIND_LINKS=http://172.17.0.1:3141/debian/" >>/root/.profile \
    && echo "fi" >>/root/.profile \
    && echo "source /root/.profile" >>/root/.zshrc \
    && pip3 install git-up \
    && true
ENTRYPOINT ["/bin/zsh", "-l"]
