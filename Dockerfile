# docker build -t goma-quri . --build-arg NB_UID=`id -u` && docker run --rm -it -v $PWD:/home/jovyan -w /home/jovyan goma-quri bash
FROM python:3.10

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

RUN mkdir /work && chown -R ${NB_UID} /work

USER ${USER}

ENV PATH ${HOME}/.local/bin:${PATH}
RUN pip3 install poetry
RUN git config --global --add safe.directory /work
