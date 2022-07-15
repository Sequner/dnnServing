from concurrent import futures
import logging

import grpc
import segmentation_pb2
import segmentation_pb2_grpc
import segmentation_models_pytorch as smp
import segmentation_model
from utils import *

CHUNK_SIZE = 16384

arriving_id = 0 # NOT SCALABLE, NEED TO CHANGE FOR NORMAL USAGE. For example, transfer filename too.

def send_image(img_id):
    with open("output/" + str(img_id) + "_out.png", 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                print("file read")
                return
            yield segmentation_pb2.UploadImageResponse(image=chunk)

class SegmentationServiceServicer(segmentation_pb2_grpc.SegmentationServiceServicer):
    def __init__(self):
        self.model = segmentation_model.SegmentationModel(f'model.bin')
    
    def Inference(self, request_iterator, context):
        global arriving_id
        img_id = receive_file(request_iterator, arriving_id)
        arriving_id += 1
        mask = self.model.inference(img_id)
        save_mask(mask, img_id)
        return send_image(img_id)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    segmentation_pb2_grpc.add_SegmentationServiceServicer_to_server(SegmentationServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("~Server Started~")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()