# Use uma imagem Python
FROM python:3.11

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y build-essential

RUN pip3 install --upgrade pip \
    pip install numpy \
    pip install scikit-learn \
    pip install matplotlib

WORKDIR /app

COPY . .

CMD ["bash"]
