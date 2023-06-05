# Dockerfile for TensorFlow GPU and FastAPI

# Use the latest TensorFlow GPU version as the base image
FROM tensorflow/tensorflow:latest-gpu

# Uncomment the line below and comment the above line to use CPU version instead
# FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Echo python version
RUN python --version

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# nltk.download('punkt')
RUN python -c 'import nltk; nltk.download("punkt")'

# Copy the entire application directory to the container
COPY . .

# Expose the port on which FastAPI will run (change as per your application)
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]
