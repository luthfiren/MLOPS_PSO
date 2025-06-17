# Use an official Python runtime as the base image.
# We choose a specific version (3.11) and a slim-buster variant for smaller image size.
# 'slim-buster' means it's based on Debian Buster, but with minimal packages installed.
FROM python:3.11-slim-buster

# Set environment variables for non-interactive operations and Python unbuffered output.
# PYTHONUNBUFFERED=1 ensures that Python's stdout and stderr are not buffered,
# which helps in seeing logs immediately in containers.
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container.
# All subsequent commands (COPY, RUN, CMD) will be executed from this directory.
# This is analogous to `cd /app` inside the container.
WORKDIR /app

# Copy the requirements.txt file into the container.
# We copy it first and install dependencies, so if only code changes,
# Docker can use a cached layer for dependency installation, speeding up builds.
COPY requirement.txt .

# Install the Python dependencies from requirements.txt.
# --no-cache-dir: Reduces image size by not storing pip's cache.
# -r requirement.txt: Installs all packages listed in the file.
RUN pip install --no-cache-dir -r requirement.txt

# Copy the entire rest of your application code into the container.
# The first '.' refers to the current directory on your local machine (where Dockerfile is).
# The second '.' refers to the WORKDIR /app inside the container.
# This will copy app.py, model/, templates/, static/, etc., into /app.
COPY . .

# Expose the port that your Gunicorn/Flask application will listen on inside the container.
# Azure App Service will map external traffic to this internal port.
# Gunicorn commonly defaults to 8000. If your app uses a different port, change this.
EXPOSE 5000

# Define the command that will be executed when the container starts.
# This is the "Startup Command" for your container.
# It tells Gunicorn to bind to all network interfaces (0.0.0.0) on port 8000,
# and to run the Flask application object named 'app' located in 'app.py'.
# The '--workers 2' argument sets the number of Gunicorn worker processes.
# Adjust the number of workers based on your App Service plan size and application needs.
CMD python ./app.py
