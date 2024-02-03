# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the necessary directories/files for inference
COPY src/inference/ /usr/src/app/src/inference/
COPY src/utils.py /usr/src/app/src/
COPY outputs/models/ /usr/src/app/models/ 

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

# Run run_inference.py when the container launches
CMD ["python", "./src/inference/run_inference.py"]
