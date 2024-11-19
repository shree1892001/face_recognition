import os
import datetime
import pickle
import cv2
from PIL import Image, ImageTk
import face_recognition
import openpyxl
import numpy as np
import tkinter as tk
from scipy.spatial.distance import cosine

import util

class App:
    def __init__(self):
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = self.create_button('Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = self.create_button('Logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = self.create_button(
            'Register New User', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = self.create_image_label()
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.cap = cv2.VideoCapture(0)  
        self.most_recent_capture_arr = None

        self.log_path = './log.txt'

        self.excel_file = 'person_logs.xlsx'
        if not os.path.exists(self.excel_file):
            self.create_excel_file()

        self.process_webcam()

    def create_excel_file(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Logs"
        ws.append(["Name", "Login Time", "Logout Time"])
        wb.save(self.excel_file)

    def create_button(self, text, color, command, fg='white'):
        return tk.Button(self.main_window, text=text, bg=color, fg=fg, command=command, font=("Arial", 14))

    def create_image_label(self):
        label = tk.Label(self.main_window)
        label.pack()
        return label

    def process_webcam(self):
        ret, frame = self.cap.read()

        if ret:
            self.most_recent_capture_arr = frame.copy()  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                name = self.compare_faces(face_encoding)
                if name is None:
                    name = "Unknown"

                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            img_with_boxes = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_with_boxes)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)

        self.webcam_label.after(20, self.process_webcam)

    def login(self):
        if self.most_recent_capture_arr is None:
            util.msg_box("Error", "No frame captured.")
            return

        name = self.recognize(self.most_recent_capture_arr)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Error', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Login Successful', f'Welcome, {name}!')
            with open(self.log_path, 'a') as f:
                f.write(f'{name},{datetime.datetime.now()},in\n')
            self.log_to_excel(name, 'Login', datetime.datetime.now())

    def logout(self):
        if self.most_recent_capture_arr is None:
            util.msg_box("Error", "No frame captured.")
            return

        name = self.recognize(self.most_recent_capture_arr)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Error', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Hasta la vista!', f'Goodbye, {name}.')
            with open(self.log_path, 'a') as f:
                f.write(f'{name},{datetime.datetime.now()},out\n')
            self.log_to_excel(name, 'Logout', datetime.datetime.now())

    def log_to_excel(self, name, action, timestamp):
        wb = openpyxl.load_workbook(self.excel_file)
        ws = wb.active
        ws.append([name, timestamp if action == 'Login' else None, timestamp if action == 'Logout' else None])
        wb.save(self.excel_file)

    def recognize(self, frame):
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return 'no_persons_found'

        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for encoding in face_encodings:
            name = self.compare_faces(encoding)
            if name:
                return name
        return 'unknown_person'

    def compare_faces(self, encoding):
        for filename in os.listdir(self.db_dir):
            if filename.endswith('.pickle'):
                with open(os.path.join(self.db_dir, filename), 'rb') as f:
                    known_encoding = pickle.load(f)
                distance = cosine(encoding, known_encoding)
                if distance < 0.6:
                    return filename.replace('.pickle', '')
        return None

    def register_new_user(self):
        if self.most_recent_capture_arr is None:
            util.msg_box("Error", "No frame captured.")
            return

        register_window = tk.Toplevel(self.main_window)
        register_window.geometry("400x200")

        tk.Label(register_window, text="Enter Name:").pack(pady=10)
        name_entry = tk.Entry(register_window)
        name_entry.pack(pady=5)

        def save_user():
            name = name_entry.get()
            if not name:
                util.msg_box("Error", "Name cannot be empty.")
                return

            embeddings = face_recognition.face_encodings(self.most_recent_capture_arr)
            if not embeddings:
                util.msg_box("Error", "No face detected in the frame.")
                return

            with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as f:
                pickle.dump(embeddings[0], f)

            util.msg_box("Success", f"User {name} registered successfully!")
            register_window.destroy()

        tk.Button(register_window, text="Save", command=save_user).pack(pady=20)

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()