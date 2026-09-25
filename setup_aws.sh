# Setup script for AWS EC2 instance (Ubuntu / Amazon Linux)
# Usage: bash setup_aws.sh

echo "=========================================="
echo ">> Setting up AWS Environment for ML Challenge"
echo "=========================================="

sudo apt-get update -y
sudo apt-get install -y python3-pip python3-dev build-essential zip unzip htop

# Install python dependencies
pip3 install --upgrade pip
pip3 install -r requirements_aws.txt

echo "Setup complete! Ready to run: python3 aws_pipeline.py"
