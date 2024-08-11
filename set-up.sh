#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# 1. Check if Docker is installed
if ! command_exists docker; then
    echo "Docker is not installed. Installing Docker..."
    
    # Install Docker based on the system type (example for Ubuntu)
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    
    # Start Docker service if needed
    sudo systemctl start docker
    sudo systemctl enable docker
else
    echo "Docker is already installed."
fi

# 2. Check if Docker Compose is installed
if ! command_exists docker-compose; then
    echo "Docker Compose is not installed. Installing Docker Compose..."
    
    # Download the latest version of Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    
    # Make it executable
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Create a symbolic link (if needed)
    sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    echo "Docker Compose has been installed."
else
    echo "Docker Compose is already installed."
fi

# # 3. Download the latest version of docker-compose.yaml for Elasticsearch
# echo "Downloading the latest version of docker-compose.yaml for Elasticsearch..."
# mkdir -p projects/folder
# curl -o projects/folder/docker-compose.yaml https://raw.githubusercontent.com/deviantony/docker-elk/master/docker-compose.yml

# # 4. Navigate to the folder
# cd projects/folder || { echo "Failed to navigate to projects/folder"; exit 1; }

# 5. Run docker-compose up
echo "Starting Docker Compose with elasticsearch-docker-8-14..."
docker-compose -f elasticsearch-docker-8-14.yml up -d

echo "Done!"

