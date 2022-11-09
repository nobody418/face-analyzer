#Importar librerias
from logging import root
import tkinter as tk
import os
from shutil import rmtree
from tkinter import messagebox as MessageBox

#Importar funciones y constantes de otros archivos
from constantes import style
from capture import proceso
from listanegra import buscarenlista, añadirlistanegra, eliminardelista

#Clase principal y primera en mostrarse
class Home(tk.Frame,):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.configure(background= style.BACKGROUND)
        self.controller=controller
        

        self.init_widgets()
#Función para saltar a la venta de eliminar usuario
    def move_to_eliminar(self):
        self.controller.show_frame(EliminarUsuario) #Envia el dato de la clase al archivo Manager
#Función para saltar a la venta de nuevo usuario
    def move_to_nuevousuario(self):
        self.controller.show_frame(NuevoUsuario)

    def reconocer(self):
        MessageBox.showinfo("Información", "Espere a que el programa inicie la cámara, aplastar Q cuando desee salir")
        os.system('python3 reconocimiento.py')

    def init_widgets(self):
        self.img = tk.PhotoImage(file="espe.png")   #Toma la imagen y la almacena en la variable
        tk.Label(self,     #Define en que lugar se colocará el frame, en este caso, en la ventana creada
        text= "Control Inteligente NRC 7622", #Texto mostrado en el label
        justify=tk.CENTER,   #Centra el label en la ventana
        **style.STYLE    #Le da los valores de la constante STYLE al estilo del label
        ).pack(  #pack define como estará situado el label
            side=tk.TOP,
            fill=tk.BOTH,   #Hace que el label creado ocupe todo el espacio disponible
            expand=True, #Hace que el label creado se expanda cuando se agranda la pantalla
            padx=22,
            pady=11
        )
        #Nuevo label
        tk.Label(self,
        image=self.img, #Coloca la imagen en el label creado
        justify=tk.CENTER,
        **style.STYLE
        ).pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )

        optionsFrame = tk.Frame(self)
        optionsFrame.configure(background = style.COMPONENT)
        optionsFrame.pack(
            side = tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )
        tk.Label(
            optionsFrame,
            text= "Bienvenido, elija una opción",
            justify=tk.CENTER,
            **style.STYLE
        ).pack(
            side=tk.TOP,
            fill=tk.X,
            padx=22,
            pady=11
        )
#Creación de botones 
        tk.Radiobutton(
                optionsFrame,   #Ubica en options frame el boton
                text="Reconocer",   #Texto del boton
                command= self.reconocer,  #llama a la función reconocer
                activebackground=style.BACKGROUND, #Le da el estilo al boton
                activeforeground=style.TEXT,   #Le da el estulo al texto del boton
                **style.STYLE
        ).pack(  #pack define como se colocará el boton en el espacio del frame
            side=tk.LEFT, #Hará que se coloque a la izquierda
            fill=tk.BOTH, #hace que se expanda en el espacio que posee
            expand=True, #Permite que se expanda si la ventana se agranda
            padx=5,
            pady=5
        )
        tk.Radiobutton(
                optionsFrame,
                text="Nuevo Usuario",
                command= self.move_to_nuevousuario,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )
        tk.Radiobutton(
                optionsFrame,
                text="Eliminar un usuario",
                command= self.move_to_eliminar,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )
        



class EliminarUsuario(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.configure(background= style.BACKGROUND)
        self.controller=controller
        self.carpetaaeliminar=tk.StringVar(self)
        self.init_widgets()

        self.ruta = 'att_faces/orl_faces/'
    def move_to_home(self):
        self.controller.show_frame(Home)
    
    def testcorrecto():
        MessageBox.showinfo("Información", "Usuario eliminado con éxito")

    def testincorrecto():
        MessageBox.showinfo("Información", "Usuario no eliminado, no existe en la base de datos")

    def eliminarcarpeta(self):
        
        añadirlistanegra(self.carpetaaeliminar.get())
        if buscarenlista(self.carpetaaeliminar.get())==True:
            EliminarUsuario.testcorrecto()
        else:    
            EliminarUsuario.testincorrecto()
        
    def init_widgets(self):
        tk.Label(self,
        text= "Escriba el nombre del usuario, después presione en Eliminar",
        justify=tk.CENTER,
        **style.STYLE
        ).pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )

        tk.Entry(
            self,
            fg="Black",
            bg="white",
            textvariable= self.carpetaaeliminar,
            justify="center",
            ).pack(
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )
        optionsFrame = tk.Frame(self)
        optionsFrame.configure(background = style.COMPONENT)
        optionsFrame.pack(
            side = tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )
        
        tk.Radiobutton(
                optionsFrame,
                text="Eliminar",
                command= self.eliminarcarpeta,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )
        
        tk.Radiobutton(
                optionsFrame,
                text="Menú Principal",
                command= self.move_to_home,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )


class NuevoUsuario(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.configure(background= style.BACKGROUND)
        self.controller=controller
        self.nombrecarpeta=tk.StringVar(self)
        self.init_widgets()

    def move_to_home(self):
        self.controller.show_frame(Home)

    def capturarrostros(self):
        try:
            if buscarenlista(self.nombrecarpeta.get()) == True:
                eliminardelista(self.nombrecarpeta.get())
                MessageBox.showinfo("Información", "Usuario añadido con éxito, no es necesario entrenar")
            else:
                proceso(self.nombrecarpeta.get())
                NuevoUsuario.testcorrecto()
        except ValueError:
            NuevoUsuario.testincorrecto()
        

        
    def testcorrecto():
        MessageBox.showinfo("Información", "Imágenes capturadas con éxito, presione en entrenar")
    
    def testincorrecto():
            MessageBox.showinfo("Información", "La toma de imágenes falló, ingrese un nombre valido")

    def entrenamiento(self):
        os.system('python3 entrenamiento.py')
        MessageBox.showinfo("Información", "Entrenamiento realizaco con éxito")

    def init_widgets(self):
        tk.Label(self,
        text= "Escriba el nombre del usuario, luego en presione capturar, finalmente, presione Entrenar",
        justify=tk.CENTER,
        **style.STYLE
        ).pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )

        tk.Entry(
            self,
            fg="Black",
            bg="white",
            justify="center",
            textvariable= self.nombrecarpeta
        ).pack(
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )
        optionsFrame = tk.Frame(self)
        optionsFrame.configure(background = style.COMPONENT)
        optionsFrame.pack(
            side = tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=22,
            pady=11
        )
        
        tk.Radiobutton(
                optionsFrame,
                text="Capturar",
                command= self.capturarrostros,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )
        tk.Radiobutton(
                optionsFrame,
                text="Entrenar",
                command= self.entrenamiento,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )
        tk.Radiobutton(
                optionsFrame,
                text="Menú Principal",
                command= self.move_to_home,
                activebackground=style.BACKGROUND,
                activeforeground=style.TEXT,
                **style.STYLE
        ).pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )
