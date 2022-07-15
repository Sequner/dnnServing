# dnnServing
gRPC-based serving system for image segmentation using U-Net

U-Net model was trained for the image segmentation of gastro-intestinal tract based on the following dataset: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation

The client-server model was implemented using gRPC. The client side sends an input image to the server. The server, then, performs inference based on pretrained model.

Server-side library requirements are in server/pretrained folder.

To start the server: ```python3 server/server.py```

To start the client: ```python3 client/client.py```
