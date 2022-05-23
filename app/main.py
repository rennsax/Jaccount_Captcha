from resnet import load_model, recognize
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

if __name__ == "__main__":
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #function to be called when mouse is clicked
    File = ""
    model = None

    def get_model():
        global model
        canvas.delete(ALL)
        model_path = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose the model.')
        model = load_model(model_path, log = False)

    def printcoords():
        global File
        canvas.delete(ALL)
        File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
        filename = ImageTk.PhotoImage(Image.open(File))
        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0,0,anchor='nw',image=filename)

    def recognize_pic():
        if model is None:
            canvas.create_text(80, 70, text="No model loaded!")
            return
        if len(File) == 0:
            canvas.create_text(80, 100, text="No picture chosen!")
            return
        canvas.create_text(50, 200, text = recognize(File, model = model), font = 20)

    Button(root,text='model',command=get_model).pack()
    Button(root,text='choose',command=printcoords).pack()
    Button(root,text='recognize',command=recognize_pic).pack()
    root.mainloop()