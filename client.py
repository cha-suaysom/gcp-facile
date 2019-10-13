import logging, grpc, time
import numpy as np
import pandas as pd
import server_tools_pb2
import server_tools_pb2_grpc
from google.protobuf import empty_pb2

PORT = '50051'
f = open("IP.txt")
IP = f.read()
if IP[-1] == '\n':
    IP = IP[:-1]
f.close()

def run_facile():
    # Get a handle to the server
    channel = grpc.insecure_channel(IP + ':' + PORT)
    stub = server_tools_pb2_grpc.MnistServerStub(channel)

    # Get a client ID which you need to talk to the server
    try:
        logging.info("Request client id from server")
        response = stub.RequestClientID(server_tools_pb2.NullParam())
    except:
        print("Connection to the server could not be established. Press enter to try again.")
        return
    client_id = response.new_id
    logging.info("Client id is " + str(client_id))

    VALSPLIT = 0.2 #0.7
    np.random.seed(5)
    Nrhs = 10000
    X = pd.read_pickle("input/X_HB.pkl")[:Nrhs]
    Y = pd.read_pickle("input/Y_HB.pkl")[:Nrhs]
    print("Sample X data", X[:10])
    print("Sample Y_data", Y[:10])

    print("Type of X converted")
    print(type(X[:10].to_records(index = False).tostring()))
    print("Type of Y converted")
    print(type(Y[:10].to_records(index = False).tostring()))

    x_input = X.to_json().encode('utf-8')
    y_input = Y.to_json().encode('utf-8')
    # Generate lots of data
    # data = np.random.rand(NUM_IMAGES, 28, 28, 1)
    #data = np.array([1,2,3])
    # data = data.tostring()

    #data should be Y

    # Send the data to the server and receive an answer
    start_time = time.time()
    logging.info("Submitting images and waiting")
    data_message = server_tools_pb2.DataMessage(client_id=client_id, x_input = x_input, y_input = y_input, batch_size=32)
    logging.info("Finish defining data message")
    response = stub.StartJobWait(data_message,100,[])
    logging.info("Finish responding")


    # Print output
    whole_time = time.time() - start_time
    print("Total time:", whole_time)
    print("Predict time:", response.infer_time)
    print("Fraction of time spent not predicting:", (1 - response.infer_time / whole_time) * 100, '%')
    print(response.prediction.decode("utf-8"))
    channel.close()




if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    #run()
    run_facile()
