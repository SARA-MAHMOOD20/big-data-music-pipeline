# Runner image for the batch/streaming scripts (Tasks 1-7). Runs attached to
# the same docker-compose network as the infra services, so it resolves
# container hostnames (namenode, datanode, mongodb, kafka, spark-master)
# natively - this sidesteps WebHDFS's datanode-redirect requiring a
# resolvable hostname, which a host-machine client can't satisfy without
# extra network configuration.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
