FROM almalinux:9

# Minimal system packages only.
# All Python packages and models live in the project directory
# because containers have no internet at runtime.
#
# IMPORTANT for COMPUTE web UI:
# Keep the RUN command on a SINGLE LINE.
# Trailing spaces after \ break Docker builds.

RUN dnf install -y wget curl git gcc gcc-c++ make which && dnf clean all

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

CMD ["/bin/bash"]
