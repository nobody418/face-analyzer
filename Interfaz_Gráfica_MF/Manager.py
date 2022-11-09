#Importar librerias
import tkinter as tk

#Importar constantes y clases de otros archivos
from constantes import style
from screens import Home, NuevoUsuario,EliminarUsuario

#Ayuda al cambio de ventanas del sistema
class Manager(tk.Tk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.title("Reconocimiento facial, expresiones y vida") #Título de la ventana
        container=tk.Frame(self)
        container.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True
        )
        container.configure(background=style.BACKGROUND)
        container.grid_columnconfigure(0,weight=1)
        container.grid_rowconfigure(0,weight=1)

        self.frames ={}
#Recibe el valor de la clase enviado desde screen y cambia de pantalla dependiendo el valor que tome F
        for F in (Home,EliminarUsuario,NuevoUsuario):
            frame = F(container,self)
            self.frames[F] =frame
            frame.grid(row =0, column=0, sticky=tk.NSEW)
        self.show_frame(Home)
    def show_frame(self, container):
        frame= self.frames[container]
        frame.tkraise()