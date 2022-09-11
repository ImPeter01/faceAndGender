from tkinter import *
from tkinter.font import BOLD, Font
from tkinter.constants import DISABLED, NORMAL
from tkinter.filedialog import askopenfilename, asksaveasfile
import cv2
from keras.models import load_model
from gender_detect import detect_image
from PIL import Image, ImageTk
import threading
from keras_preprocessing.image import img_to_array
import numpy as np



class UI:
    def __init__(self, root):
        self.check_camera = False
        self.classes = ['man', 'woman']
        global cam
        global frame
        global ret
        global photo
        global img_save
        global status_predict
        self.root = root
        self.root.title("GENDER DETECT")  # title of the GUI window
        self.root.maxsize(900, 600)  # specify the max size the window can expand to
        self.StartUI()
        self.root.config(bg="skyblue")
    def StartUI(self):
        self.model = load_model('gender_detect.model')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.left_frame = Frame(self.root, width=150, height=400, bg='grey').grid(row = 0, column  = 0)

        self.extend_frame = Frame(self.left_frame, width=150, height=100, bg='grey').grid(row = 0, column  = 0,sticky = S,padx=5, pady=5 )
        self.menu_frame = Frame(self.left_frame, width=150, height=300, bg='grey').grid(row=0, column=0, sticky=N,padx=5, pady=5)

        self.right_frame = Frame(self.root, width=650, height=400, bg='grey').grid(row = 0, column  = 1,padx=5, pady=5)
        # left frame

        #self.tool_bar_bot = Frame(self.bot_frame, width=180, height=20)
        #self.tool_bar_bot.grid(row=1, column=0, padx=5, pady=5)
        self.bold12 = Font(self.root, size=12, weight=BOLD)
        Label(self.menu_frame, text = "MODE USING",padx=5, pady=5, bg = "grey", fg = "white", font=self.bold12).place(x=20,y = 10)
        Label(self.extend_frame, text="EXTEND MODE", padx=5, pady=5,bg = "grey", fg = "white", font=self.bold12).place(x=13,y=250)
        # create button

        self.btn_using_img = Button(master =self.menu_frame, text="MODE IMAGE", width=15, height=1,
                                    command=self.useimage_func,activebackground='skyblue',bd=3,relief=RAISED).place(x=22, y= 45)
        self.btn_realtime = Button(master=self.menu_frame, text="MODE REALTIME", width=15, height=1,
                                    command=self.userealtime_func, activebackground='skyblue', bd=3, relief=RAISED).place(x=22, y=80)
        self.btn_saveimage = Button(master=self.menu_frame, text="MODE SAVE IMAGE", width=15, height=1,
                                   command=self.save_img, activebackground='skyblue', bd=3,
                                   relief=RAISED, state = "disabled")
        self.btn_saveimage.place(x=22, y=115)
        self.btn_predict = Button(master=self.menu_frame, text="PREDICT IMAGE", width=15, height=1,
                                    command=self.predict, activebackground='skyblue', bd=3,
                                    relief=RAISED, state = "disabled" )
        self.btn_predict.place(x=22, y=280)
        self.btn_snapshot = Button(master=self.menu_frame, text="SNAPSHOT IMAGE", width=15, height=1,
                                    command=self.snapshot, activebackground='skyblue', bd=3,
                                    relief=RAISED,state = "disabled")
        self.btn_snapshot.place(x=22, y=315)

        self.label = Label(self.right_frame,text = "ĐỀ TÀI \n ỨNG DỤNG DEEP LEARNING VÀO XỬ LÍ ẢNH NHẬN DIỆN FACE AND GENDER. \n \n GVHD: BÙI CÔNG TUẤN \n --------- \n \n SVTH 1: HUỲNH TẤN PHONG \n SVTH 2: PHẠM PHÚ THÀNH  ",
                           height=20, font = self.bold12)
        self.label.grid(row=0, column=1,sticky=W + E + N + S, padx=5, pady=5)

    def useimage_func(self):
        print("use image")
        #self.btn_predict['state']='!disabled'
        #self.btn_snapshot['state']='disabled'
        self.btn_predict['state'] = NORMAL
        self.btn_snapshot['state'] = DISABLED
        self.btn_saveimage['state'] = NORMAL

        if self.check_camera == True:
            self.cam.release()
            self.check_camera = False

        self.filename = askopenfilename(
            filetypes=(("jpg file", "*.jpg"), ("png file", '*.png'), ("All files", " *.* "),))
        self.photo = Image.open(self.filename)
        if (self.photo.width > 650 and  self.photo.height > 400) or (self.photo.width > 650) or (self.photo.height > 400):
            self.photo = self.photo.resize((650, 400))
        self.photo = ImageTk.PhotoImage(self.photo)
        self.label.configure(image=self.photo,width=650, height=400)
        self.label.image = self.photo

    def userealtime_func(self):
        print("use realtime")
        #self.btn_predict.state = ['disabled']
        #self.btn_snapshot.state = ['!disabled']
        self.btn_predict['state'] = DISABLED
        self.btn_snapshot['state'] = NORMAL
        self.btn_saveimage['state'] = NORMAL
        self.check_camera = True
        if self.check_camera:
            self.cam = cv2.VideoCapture(0)
            while self.check_camera:
                if self.check_camera == False:
                    self.cam.release()
                self.ret, self.frame = self.cam.read()
                frame_cpy = self.frame.copy()
                frame_cpy = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2GRAY)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                face = self.face_cascade.detectMultiScale(
                    frame_cpy,
                    scaleFactor=1.2,
                    minNeighbors=2,
                    minSize=(30, 30))

                # loop through detected faces
                for idx, f in enumerate(face):
                    print("processing detect face")
                    # get corner points of face rectangle
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    print("startX", startX)
                    print("startY", startY)
                    print("endX", endX)
                    print("endY", endY)

                    # draw rectangle over face
                    cv2.rectangle(self.frame, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)

                    # crop the detected face region
                    face_crop = self.frame[startY:(startY + endY), startX:(startX + endX)]
                    print("shape[0]", face_crop.shape[0])
                    print("shape[1]", face_crop.shape[1])
                    print("shape[2]", face_crop.shape[2])

                    if (face_crop.shape[0]) < 30 or (face_crop.shape[1]) < 30:
                        continue

                    # draw rectangle over face
                    cv2.rectangle(self.frame, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)

                    # preprocessing for gender detection model
                    face_crop = cv2.resize(face_crop, (96, 96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # apply gender detection on face
                    conf = self.model.predict(face_crop,
                                              batch_size=32)  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                    # get label with max accuracy
                    idx = np.argmax(conf)
                    print("result data:", idx)
                    label = self.classes[idx]
                    percents = conf[0][idx] * 100

                    # label = conf[idx] * 100 + "," + label
                    label = "{gender},{percent}".format(gender=label, percent="{:.2f}%".format(percents))
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    # write label and confidence above face rectangle
                    cv2.putText(self.frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                self.img_save = self.frame;
                img_update = ImageTk.PhotoImage(Image.fromarray(self.frame))
                self.label.configure(image=img_update, width=650, height=400)
                self.label.image = img_update
                self.label.update()


    def save_img(self):
        print("use save image")
        #self.btn_predict.state = ['disabled']
        #self.btn_snapshot.state = ['disabled']
        self.btn_predict['state'] = DISABLED
        self.btn_snapshot['state'] = DISABLED
        file = asksaveasfile(mode='w', defaultextension=".png")
        if file:
            self.img_save = cv2.cvtColor(self.img_save, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file.name, self.img_save)
            return True
        return False



    def predict(self):
        self.temp = self.filename
        self.filename.replace("/", "/")
        print(self.filename)

        self.photo = cv2.imread(self.filename, 1)
        self.photo = detect_image(self.photo, self.model, self.face_cascade)
        self.photo = cv2.cvtColor(self.photo, cv2.COLOR_BGR2RGB)
        self.img_save = self.photo
        self.photo = Image.fromarray(self.photo)
        if (self.photo.width > 650 and self.photo.height > 400) or (self.photo.width > 650) or (
                self.photo.height > 400):
            self.photo = self.photo.resize((650, 400))
        self.photo = ImageTk.PhotoImage(self.photo)
        self.label.configure(image=self.photo, width=650, height=400)
        self.label.image = self.photo
        # self.file = asksaveasfile(mode='w', defaultextension=".png")
        # cv2.imwrite(self.file.name,self.photo)
        print("use predited")

    def snapshot(self):
        print("use snapshot")
        if self.check_camera == True:
            self.cam.release()
            self.check_camera = False
        self.photo = self.frame
        self.photo = Image.fromarray(self.photo)
        self.photo = ImageTk.PhotoImage(self.photo)
        self.label.configure(image=self.photo, width=650, height=400)
        self.label.image = self.photo

# Create mode using

