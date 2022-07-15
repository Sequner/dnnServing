import logging
import os
from PIL import ImageTk, Image
import time

import grpc
import segmentation_pb2
import segmentation_pb2_grpc
from utils import *

import tkinter as tk

CHUNK_SIZE = 16384
count = 0
start_time = 0
end_time = 0

def send_image(image_path):
    with open(image_path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                return
            yield segmentation_pb2.UploadImageRequest(image=chunk)

def main_page():
    root = tk.Tk()
    root.title('Main page')
    root.geometry("200x200+700+300")
    def start():
        channel = grpc.insecure_channel('localhost:50051')
        stub = segmentation_pb2_grpc.SegmentationServiceStub(channel)
        root.withdraw()
        scan_page = tk.Toplevel()
        scan_page.geometry("700x600+700+200")
        scan_page.title('Scan page')
        def quit():
            channel.close()
            scan_page.destroy()
            root.deiconify()

        def run():
            global count
            global start_time
            global end_time
            ls = sorted(filter(lambda x: os.path.isfile(os.path.join("input", x)),
                        os.listdir("input")))
            if ls:
                if count == 0:
                    start_time = time.time()
                i = ls[0]
                receive_file(stub.Inference(send_image("input/"+i)))
                os.remove("input/"+i)
                img = Image.open("output.png")
                img = img.resize((500,500), Image.ANTIALIAS)
                my_img = ImageTk.PhotoImage(img)
                label.configure(image=my_img)
                label.image = my_img
                label.pack()
                count += 1
                if count == 99:
                    end_time = time.time()
                    print("%s seconds" % ((end_time-start_time) / 100))
            label.after(10, run)
        label = tk.Label(scan_page)
        label.place(x=50, y=20)
        s_img = tk.PhotoImage(file="icons/blue.png", height=20, width=20)
        stomach_label = tk.Label(scan_page, text = "  stomach", image=s_img, compound='left')
        stomach_label.image = s_img
        stomach_label.pack(anchor='nw')
        l_img = tk.PhotoImage(file="icons/green.png", height=20, width=20)
        l_label = tk.Label(scan_page, text = "  large bowel", image=l_img, compound='left')
        l_label.image = l_img
        l_label.pack(anchor='nw')
        s_img = tk.PhotoImage(file="icons/red.png", height=20, width=20)
        s_label = tk.Label(scan_page, text = "  small bowel", image=s_img, compound='left')
        s_label.image = s_img
        s_label.pack(anchor='nw')

        
        scan_page.protocol("WM_DELETE_WINDOW", quit)
        label.after(1, run)

    start_button = tk.Button(root, text="Start scan", command=start)
    start_button.pack(ipady=10, side="top", fill='x', pady=50)
    root.mainloop()

def scan_page():
    root = tk.Tk()
    root.mainloop()

if __name__ == '__main__':
    logging.basicConfig()
    main_page()
    # run()