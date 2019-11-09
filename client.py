import logging
import grpc
import time
import numpy as np
import pandas as pd
import server_tools_pb2
import server_tools_pb2_grpc
from google.protobuf import empty_pb2

PORT = '50051'

def run_facile(channel, data_send, num_data_send, batch_size):
    start_time = time.time()
    # Get a client ID which you need to talk to the server
    stub = server_tools_pb2_grpc.FacileServerStub(channel)
    try:
        #logging.info("Request client id from server")
        response = stub.RequestClientID(server_tools_pb2.NullParam())
    except BaseException:
        print(
             """Connection to the server could not be established.
             Press enter to try again.""")
        return
    client_id = response.new_id
    #logging.info("Client id is " + str(client_id))
    finish_time = time.time()-start_time

    #print("Time establishing server connections ", finish_time)
    # Send the data to the server and receive an answer
    start_time = time.time()
    #logging.info("Number tested is " + str(num_data_send))
    #logging.info("Submitting files and waiting")
    data_message = server_tools_pb2.DataMessage(
        client_id=client_id, data=data_send, batch_size=batch_size)
    response = stub.StartJobWait(data_message, 100, [])

    # Print output
    whole_time = time.time() - start_time
    #print("Whole time:", whole_time)
    #print("Predict time:", response.infer_time)
    #print("Fraction of time spent not predicting:",
    #      (1 - response.infer_time / whole_time) * 100, '%')
    A = np.frombuffer(response.prediction,dtype = np.float32)
    print(list(np.frombuffer(response.prediction,dtype = np.float32))[:10])
    channel.close()
    return response.infer_time_CPU, response.infer_time_GPU


def setup_server(host_IP):
    start_time = time.time()
    options = [('grpc.max_receive_message_length', 500*1024*1024 )]
    channel = grpc.insecure_channel(host_IP + ':' + PORT, options = options)
    
    finish_time = time.time()-start_time
    #print("Time defining server stub is ", finish_time)
    return channel

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--IP', type=str, default="localhost")
    parser.add_argument('--num_send', type=int, default="localhost")

    args = parser.parse_args()
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)

    start_time = time.time()
    read_rec_hit = pd.read_pickle("input/X_HB.pkl") #[:args.num_send]
    read_rec_hit.drop(['PU', 'pt'], 1, inplace=True)
    mu,std = np.mean(read_rec_hit, axis=0), np.std(read_rec_hit, axis=0)
    print("mu is ", mu)
    print("std is ", std)
    read_rec_hit = (read_rec_hit-mu)/std
    read_rec_hit = read_rec_hit[:args.num_send]
    print(len(read_rec_hit))
    finish_time = time.time()-start_time
    print("Time reading data from local file and preprocessing (pkl->pandas) is ", finish_time)

    start_time = time.time()
    compressed_data = read_rec_hit.to_json().encode('utf-8')
    finish_time = time.time()-start_time
    print("Time reading data from local file (pandas->bytes) is ", finish_time)
    #num_run = 5
    #time_average = 0
    #for i in range(num_run):
    #    time_average += run_facile(setup_server(args.IP), compressed_data, args.num_send)
    #print(time_average/num_run)
    for i in 2**np.arange(8,17):
       cpu_time, gpu_time = run_facile(setup_server(args.IP), compressed_data, args.num_send, i)
       print(cpu_time,gpu_time)    
#print(time_average/10)
    #num_run = 1
    #time_average = 0
    #for i in range(num_run):
    #    time_average += run_facile(setup_server(args.IP), compressed_data, args.num_send)
    #print(time_average/num_run)

