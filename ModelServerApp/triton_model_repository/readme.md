### Running the server

To run Triton Inference Server:

docker run -it -v triton_model_repository/:/models  -p5000:8000 -p5001:8001 -p5002:8002 nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --strict-model-config=False --model-repository=/models

Check server health:

curl -v localhost:<port>/v2/health/ready
