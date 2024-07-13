# dockerize_braindiary_rag

# Brain Diary RAG Application README


### Prerequisites

- Docker installed on your machine.
- Python installed on your machine.

### Installation Steps

#### Step 1: Clone the Repository

First, clone this repository to your local machine using git clone

#### Step 2: Start Docker Service

Ensure Docker is running on your machine. You can start Docker if it's not already running.

#### Step 3: Pull Docker Images

Pull the necessary Docker images for Ollama and MongoDB using docker pull command


#### Step 4: Build Docker Image

Build the Docker image for the Brain Diary RAG application:

'''bash 
docker build . -t braindiary_rag
'''

Verify the Docker image was successfully built by listing the available images:

'''bash 
docker images
'''


#### Step 5: Run Docker Compose

Start the application by running Docker Compose:

'''bash
docker-compose up
'''


This command will start both the Ollama and MongoDB containers.
Check the status of the containers:
'''bash 
docker-compose ps
'''


#### Step 6: Download Required Model

Access the Ollama container and download the required model:

'''bash 
docker exec -it <container_name> ollama run phi3:instruct
'''

#### Step 7: Access Streamlit Application

Finally, open your web browser and navigate to:
http://localhost:8501
This will load the Streamlit application, allowing you to interact with the Brain Diary RAG application.


#### Note : To run it in Colab upload ui.py, rag_pdf.py, rag_web.py, requirements.txt to colab and follow below steps 

 Use attached notebook, upload to colab and refer this blog https://medium.com/@mauryaanoop3/running-ollama-on-google-colab-free-tier-a-step-by-step-guide-9ef74b1f8f7a

Also it requires setting up ngrok to run streamlit app






