# Use an official Python 3.10 runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy only the necessary directories/files for training
COPY src/train/ /usr/src/app/src/train/
COPY requirements.txt /usr/src/app/
COPY src/utils.py /usr/src/app/src/
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run train.py when the container launches
CMD ["python", "./src/train/train.py"]
