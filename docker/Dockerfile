FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&  apt-get install -y \
    git \
    build-essential \
    vim \
    openssh-server \
    python3-opencv \
    ca-certificates \
    python3-dev \
    python3-pip \
    wget \
    ninja-build \
    mesa-common-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    xauth

RUN sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config

RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN pip3 install numpy==1.23.4 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN git clone https://github.com/cherubicXN/hawp.git /root/hawp

RUN  echo 'root:root' | chpasswd

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

RUN service ssh start

CMD ["/usr/sbin/sshd","-D"]
