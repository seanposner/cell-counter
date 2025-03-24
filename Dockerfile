# Use an official Ubuntu base image
FROM ubuntu:24.04

# Set working directory
WORKDIR /app

# Copy requirements
COPY . .

# Install basic dependencies, Python3, pip
RUN apt-get update && \
    apt-get install -y $(cat apt-requirements.txt)

# Install Python packages
RUN pip3 install --no-cache-dir -r pip-requirements.txt

# When the container starts, run Cell-Counter.py
ENTRYPOINT ["python3", "Cell-Counter.py"]
