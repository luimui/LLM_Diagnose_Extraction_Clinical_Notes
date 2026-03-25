4# Use Miniconda base image
FROM continuumio/miniconda3:latest

# Set working directory inside container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Create a new conda environment named 'myenv' and install pip packages
RUN conda create --name myenv python=3.9.24 && \
    conda run -n myenv pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Use conda run to start Streamlit
CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
