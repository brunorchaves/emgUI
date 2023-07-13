import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

gestureSelected = "none"
gesture_images = {
    "Pointer": "D:\Documentos\GitHub\emgUI\customTkinter\imgs\pointer.png",
    "Rock": "D:\Documentos\GitHub\emgUI\customTkinter\imgs\imgrock.png",
    "Open": "D:\Documentos\GitHub\emgUI\customTkinter\imgs\open.png",
    "Spock": "D:\Documentos\GitHub\emgUI\customTkinter\imgs\spock.png",
    "Thumbs Up": "D:\Documentos\GitHub\emgUI\customTkinter\imgs\imgthumbsup.png",
    # Add more gestures and their corresponding image paths
}


class ctkApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.geometry("1200x600")
        self.root.title("EMG dataset-maker")
        self.root.update()

        # Left Corner frame for "Gestures types" text placement
        self.frame = ctk.CTkFrame(master=self.root,
                                  height= self.root.winfo_height()*0.95,
                                  width = self.root.winfo_width()*0.66)
        self.frame.place(relx=0.04, rely=0.025)
        
        # Left Corner text for "Gestures types"
        self.label1  = ctk.CTkLabel(master=self.frame, text="Gestures", font=('Roboto', 24))
        self.label1.pack(pady=12, padx= 10)

        self.frame = ctk.CTkFrame(master=self.root,
                                  height= self.root.winfo_height()*0.95,
                                  width = self.root.winfo_width()*0.66)
        self.frame.place(relx=0.2, rely=0.025)
        
        self.label2 = ctk.CTkLabel(master=self.frame, text="Collect gesture data", font=('Roboto', 24))
        self.label2.pack(pady=12, padx= 10)

        self.label3 = ctk.CTkLabel(master=self.root, text="Gesture selected: "+gestureSelected, font=('Roboto', 12))
        self.label3.place(relx=0.2,rely=0.15)
        # self.label.pack(pady=12, padx= 10)
        self.image_label = ctk.CTkLabel(master=self.root)
        self.image_label.place(relx=0.2, rely=0.25)
        # image_path = gesture_images.get("Pointer")
        # image = Image.open(image_path)
        # photo = ImageTk.PhotoImage(image)
        # self.image_label.configure(image=photo)
        # self.image_label.image = photo  # Keep a reference to avoid garbage collection
   

        self.button1 = ctk.CTkButton(
                                master=self.root,
                                text="Pointer",
                                width=150,
                                height=25,
                                command=lambda: self.writeGesureName(self.button1.cget("text")))
        self.button1.place(relx=0.0,rely=0.25)
        self.button2 = ctk.CTkButton(master = self.root,
                               text="Rock",
                               width=150,
                               height=25,
                               command=lambda: self.writeGesureName(self.button2.cget("text")))
        self.button2.place(relx=0.0,rely=0.35)
        self.button3 = ctk.CTkButton(master = self.root,
                               text="Open",
                               width=150,
                               height=25,
                               command=lambda: self.writeGesureName(self.button3.cget("text")))
        self.button3.place(relx=0.0,rely=0.45)
        self.button4 = ctk.CTkButton(master = self.root,
                               text="Spock",
                               width=150,
                               height=25,
                               command=lambda: self.writeGesureName(self.button4.cget("text")))
        self.button4.place(relx=0.0,rely=0.55)
        self.button5 = ctk.CTkButton(master = self.root,
                               text="Thumbs Up",
                               width=150,
                               height=25,
                               command=lambda: self.writeGesureName(self.button5.cget("text")))
        self.button5.place(relx=0.0,rely=0.65)
        self.plot_emg_graph()
        # self.checkbox= ctk.CTkCheckBox(master= self.frame, text="Remember Me")
        # self.checkbox.pack(pady=12,padx= 10)
        self.root.mainloop()
        
    def login(self):
            print("Test") 

    gestureLabel= " "      
    def writeGesureName(self, gestureLabel):
        gestureSelected = gestureLabel
        self.label3.configure(text="Gesture selected: " + gestureSelected)
        # Load and display the image
        image_path = gesture_images.get(gestureSelected)
        if image_path:
            image = Image.open(image_path)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference to avoid garbage collection
        else:
            self.image_label.configure(image=None)
    def update_window(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(5,3)
        global x,y,s,c
        x,y,s,c = np.random.rand(4,int(100))
        ax.scatter(x,y,s*5,c)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        canvas = FigureCanvasTkAgg(fig,master=self.root)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.33, rely=0.025)
        self.root.update()
        
    def update_surface(self,other):
        fig, ax = plt.subplots()
        fig.set_size_inches(11,5.3)
        ax.scatter(x,y,s*5,c)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        canvas = FigureCanvasTkAgg(fig,master=self.root)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.33, rely=0.025)
        self.root.update()


    def plot_emg_graph(self):
        
        # Generating sample data for 8 channels
        num_channels = 8
        duration = 5  # seconds
        sampling_rate = 1000  # samples per second
        num_samples = duration * sampling_rate

        # Generate random EMG data for each channel
        np.random.seed(42)  # For reproducibility
        emg_data = np.random.randn(num_channels, num_samples)

        # Generate time array
        time = np.arange(num_samples) / sampling_rate

        # Define labels for each channel
        channel_labels = ['Channel A', 'Channel B', 'Channel C', 'Channel D',
                        'Channel E', 'Channel F', 'Channel G', 'Channel H']
        # Define colors for each channel
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']


        fig, axs = plt.subplots(num_channels, 1, figsize=(6, 4), sharex=True)
        # fig.set_size_inches(11,5.3)
        # Plotting each channel separately
        for i in range(num_channels):
            axs[i].plot(time, emg_data[i], color=colors[i])  # Set color for each channel
            axs[i].set_ylabel("Volts",fontsize = 6)
            axs[i].grid(True)
            # Position the channel label inside the graph
            x_label = time[-1] * 0.95  # x-coordinate for the label
            y_label = np.max(emg_data[i]) * 0.9  # y-coordinate for the label

        # Set common x-axis label and title
        axs[-1].set_xlabel("Time (s)")
        fig.suptitle("EMG Live signal")
        # Adjust the spacing between subplots
        fig.tight_layout()
        # Create a canvas for the plot
        fig.subplots_adjust(left=0.15, right=1, bottom=0.15, top=0.9, wspace=0, hspace=0)
        canvas = FigureCanvasTkAgg(fig,master=self.root)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.4, rely=0.025)
        self.root.update()



if __name__ == "__main__":        
    CTK_Window = ctkApp()
    # Update the window
    # root.update()