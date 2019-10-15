# gcp-tensorRT

This repository sets up the as-a-service server on Google Cloud Platform (gcp) to perform deep neural network. In addition to the GPU available in gcp, we also plan to integrate tensorRT to this server. The steps necessary to set up the server can be found in [mnist-grpc](https://github.com/JackDinsmore/mnist-server-grpc/tree/tpu) 

If the server is already running, then the users have to perform the following steps

1. Install the necessary packages with `pip install -r requirements.txt` (I strongly suggest running from gcp shell because they already have most of these packages)
2. Put the necessary input files as `input/X_HB.pkl`, `input/Y_HB.pkl`
3. Compile the proto file with `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. server-tools.proto`
4. Run `python client.py --IP="ip-address"` where `ip-address` is the location of the server (we use target port of 50051 by default) or `localhost` for local testing.

