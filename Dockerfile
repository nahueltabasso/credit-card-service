FROM python:3.9.6
ENV N_PROCESSES=1
ENV DIRECTORY=/opt/project/credit-card-service

RUN mkdir -p ${DIRECTORY}

WORKDIR ${DIRECTORY}

COPY . ${DIRECTORY}
RUN apt-get update && \
    apt-get install -y wget curl git vim iputils-ping gcc libpq-dev software-properties-common locales libgl1-mesa-glx && \
    apt-get upgrade -y && \
    ln -f -s /usr/bin/python3 /usr/bin/python && \
    echo "es_AR.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    locale -a && \
    export LC_ALL="es_AR.utf8" && \
    export LC_CTYPE="es_AR.utf8" && \
    locale -a
RUN python -m pip install --upgrade pip
# Install dependencies from requirements
RUN pip install -r requirements.txt
# Install GroundingDINO
RUN pip install git+https://github.com/IDEA-Research/GroundingDINO.git@df5b48a3efbaa64288d8d0ad09b748ac86f22671
RUN mkdir weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN pip install -e .

CMD cd ${DIRECTORY}/src/demo && python gradio_ui.py
