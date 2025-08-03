import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from neuralNetwork.utilities.ImageProcessor import ImageProcessor

class FaceRecognitionApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        ctk.set_window_scaling(1.0)
        ctk.set_widget_scaling(1.1)

        self.root = ctk.CTk()
        self.root.title("Reconocimiento Facial - Red Neuronal")
        self.root.geometry("900x700")
        
        self.root.resizable(True, True)
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.current_image = None
        self.image_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.root,
            width=850,
            height=650
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        title_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="Sistema de Reconocimiento Facial",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        main_frame = ctk.CTkFrame(self.scrollable_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        load_section = ctk.CTkFrame(main_frame)
        load_section.pack(fill="x", padx=20, pady=10)
        
        load_label = ctk.CTkLabel(
            load_section,
            text="Cargar Imagen",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        load_label.pack(pady=10)
        
        self.load_button = ctk.CTkButton(
            load_section,
            text="Seleccionar Imagen",
            command=self.load_image,
            width=200,
            height=40
        )
        self.load_button.pack(pady=10)
        
        preview_frame = ctk.CTkFrame(main_frame)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        preview_label = ctk.CTkLabel(
            preview_frame,
            text="Previsualización",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        preview_label.pack(pady=10)
        
        self.image_label = ctk.CTkLabel(
            preview_frame,
            text="No hay imagen cargada",
            width=400,
            height=300
        )
        self.image_label.pack(pady=10)
        
        self.info_label = ctk.CTkLabel(
            preview_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.info_label.pack(pady=5)
        
        process_frame = ctk.CTkFrame(main_frame)
        process_frame.pack(fill="x", padx=20, pady=10)
        
        self.process_button = ctk.CTkButton(
            process_frame,
            text="Procesar con Red Neuronal",
            command=self.process_image,
            width=250,
            height=50,
            state="disabled"
        )
        self.process_button.pack(pady=15)
        
        result_frame = ctk.CTkFrame(main_frame)
        result_frame.pack(fill="x", padx=20, pady=10)
        
        result_title = ctk.CTkLabel(
            result_frame,
            text="Resultado del Reconocimiento",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        result_title.pack(pady=10)
        
        self.result_label = ctk.CTkLabel(
            result_frame,
            text="Esperando procesamiento...",
            font=ctk.CTkFont(size=14),
            width=400,
            height=60
        )
        self.result_label.pack(pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            result_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.confidence_label.pack(pady=5)
        
        # Barra de progreso (opcional)
        self.progress_bar = ctk.CTkProgressBar(result_frame)
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.set(0)
        
    def load_image(self):
        """Cargar imagen desde el sistema de archivos"""
        file_types = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos los archivos", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=file_types
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.display_image(file_path)
                self.process_button.configure(state="normal")
                self.result_label.configure(text="Imagen cargada. Lista para procesar.")
                self.confidence_label.configure(text="")
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def display_image(self, image_path):
        """Permite mostrar una vista previa de la imagen"""
        try:
            pil_image = Image.open(image_path)
            self.current_image = pil_image.copy()
            
            width, height = pil_image.size
            file_size = os.path.getsize(image_path) / 1024  # KB
            
            self.info_label.configure(
                text=f"Dimensiones: {width}x{height} | Tamaño: {file_size:.1f} KB"
            )
            
            # Redimensionar para mostrar (manteniendo aspecto)
            display_size = (300, 250)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convertir a formato compatible con CustomTkinter
            photo = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=pil_image.size
            )
            
            # Actualizar el label con la imagen
            self.image_label.configure(image=photo, text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar la imagen: {str(e)}")
    
    def process_image(self):
        """Procesa la imagen con la red neuronal"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "Por favor, carga una imagen primero.")
            return
        
        try:
            # Mostrar progreso
            self.progress_bar.set(0.2)
            self.result_label.configure(text="Procesando imagen...")
            self.root.update()
            
            # Simular procesamiento de la red neuronal
            # AQUÍ ES DONDE INTEGRARÍAS TU RED NEURONAL
            result, confidence = self.neural_network_prediction()
            
            self.progress_bar.set(0.8)
            self.root.update()
            
            # Mostrar resultado
            if result:
                self.result_label.configure(
                    text="✅ ¡ROSTRO RECONOCIDO!",
                    text_color="green"
                )
            else:
                self.result_label.configure(
                    text="❌ Rostro no reconocido",
                    text_color="red"
                )
            
            self.confidence_label.configure(
                text=f"Nivel de confianza: {confidence:.1f}%"
            )
            
            self.progress_bar.set(1.0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
            self.progress_bar.set(0)
    
    def neural_network_prediction(self):
        """
        MÉTODO PARA INTEGRAR TU RED NEURONAL
        
        Aquí debes reemplazar este código con tu red neuronal real.
        Este es solo un ejemplo que simula el comportamiento.
        """
        
        # Preprocesar imagen para tu red neuronal
        # Ejemplo de preprocesamiento básico:
        image_array = np.array(self.current_image)
        
        # Convertir a RGB si es necesario
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Redimensionar a las dimensiones que espera tu red (ejemplo: 224x224)
        # processed_image = cv2.resize(image_array, (224, 224))
        
        # Normalizar
        # processed_image = processed_image / 255.0
        
        # AQUÍ LLAMAS A TU RED NEURONAL:
        # prediction = tu_modelo.predict(processed_image)
        # confidence = float(prediction[0]) * 100
        # is_recognized = prediction[0] > 0.5  # umbral de decisión
        
        # SIMULACIÓN (reemplazar con tu código real):
        import random
        confidence = random.uniform(60, 95)
        is_recognized = confidence > 75
        
        return is_recognized, confidence
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

# Función principal
def main():
    app = FaceRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()