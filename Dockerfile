# Use a lightweight base image with Python
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0
# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application files into the container
COPY app /app

# Expose the port your app runs on
EXPOSE 8000

# Set the default command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
