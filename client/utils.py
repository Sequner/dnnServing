def receive_file(image_iterator):
    with open("output.png", 'wb') as f:
        for chunk in image_iterator:
            f.write(chunk.image)