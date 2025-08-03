import customtkinter as ctk
from tkinter import filedialog, messagebox, font
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys

# Configuración para Windows - Arreglar pixelación
if sys.platform.startswith('win'):
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

# Función para obtener fuentes seguras multiplataforma
def get_safe_font_family():
    """Obtiene una fuente que funcione en el sistema actual"""
    try:
        # Crear ventana temporal para acceder a las fuentes
        import tkinter as tk
        temp_root = tk.Tk()
        temp_root.withdraw()
        
        available_fonts = list(font.families())
        temp_root.destroy()
        
        # Lista de fuentes preferidas en orden de prioridad
        preferred_fonts = [
            "DejaVu Sans", "Liberation Sans", "FreeSans",  # Linux comunes
            "Arial", "Helvetica", "Ubuntu", "Cantarell",   # Alternativas
            "sans-serif", "TkDefaultFont"                   # Fallbacks
        ]
        
        # Buscar la primera fuente disponible
        for font_name in preferred_fonts:
            if font_name in available_fonts:
                return font_name
        
        # Si no encuentra ninguna, usar la primera disponible
        return available_fonts[0] if available_fonts else "TkDefaultFont"
        
    except Exception as e:
        print(f"Error detectando fuentes: {e}")
        return "TkDefaultFont"

class FaceRecognitionApp:
    def __init__(self):
        # Configuración de CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configuración de escalado para mejorar la calidad visual
        ctk.set_widget_scaling(1.0)  # Escalado de widgets
        ctk.set_window_scaling(1.0)  # Escalado de ventana
        
        # Detectar fuente segura para el sistema
        self.safe_font = get_safe_font_family()
        print(f"Usando fuente: {self.safe_font}")
        
        # Ventana principal
        self.root = ctk.CTk()
        self.root.title("Reconocimiento Facial - Red Neuronal")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Configurar peso de las columnas y filas para el redimensionamiento
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Variables
        self.current_image = None
        self.image_path = None
        
        # Crear la interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Crear un frame scrollable que contenga todo el contenido
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.root,
            width=750,
            height=650
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título principal
        title_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="Sistema de Reconocimiento Facial",
            font=ctk.CTkFont(family=self.safe_font, size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Frame principal (ahora dentro del scrollable frame)
        main_frame = ctk.CTkFrame(self.scrollable_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Sección de carga de imagen
        load_section = ctk.CTkFrame(main_frame)
        load_section.pack(fill="x", padx=20, pady=10)
        
        load_label = ctk.CTkLabel(
            load_section,
            text="Cargar Imagen",
            font=ctk.CTkFont(family=self.safe_font, size=16, weight="bold")
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
        
        # Frame para previsualización
        preview_frame = ctk.CTkFrame(main_frame)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        preview_label = ctk.CTkLabel(
            preview_frame,
            text="Previsualización",
            font=ctk.CTkFont(family=self.safe_font, size=16, weight="bold")
        )
        preview_label.pack(pady=10)
        
        # Label para mostrar la imagen
        self.image_label = ctk.CTkLabel(
            preview_frame,
            text="No hay imagen cargada",
            width=400,
            height=300
        )
        self.image_label.pack(pady=10)
        
        # Información de la imagen
        self.info_label = ctk.CTkLabel(
            preview_frame,
            text="",
            font=ctk.CTkFont(family=self.safe_font, size=12)
        )
        self.info_label.pack(pady=5)
        
        # Frame para procesamiento
        process_frame = ctk.CTkFrame(main_frame)
        process_frame.pack(fill="x", padx=20, pady=10)
        
        # Botón para procesar
        self.process_button = ctk.CTkButton(
            process_frame,
            text="Procesar con Red Neuronal",
            command=self.process_image,
            width=250,
            height=50,
            state="disabled"
        )
        self.process_button.pack(pady=15)
        
        # Frame para resultados
        result_frame = ctk.CTkFrame(main_frame)
        result_frame.pack(fill="x", padx=20, pady=10)
        
        result_title = ctk.CTkLabel(
            result_frame,
            text="Resultado del Reconocimiento",
            font=ctk.CTkFont(family=self.safe_font, size=16, weight="bold")
        )
        result_title.pack(pady=10)
        
        # Label para mostrar el resultado
        self.result_label = ctk.CTkLabel(
            result_frame,
            text="Esperando procesamiento...",
            font=ctk.CTkFont(family=self.safe_font, size=14),
            width=400,
            height=60
        )
        self.result_label.pack(pady=10)
        
        # Indicador de confianza
        self.confidence_label = ctk.CTkLabel(
            result_frame,
            text="",
            font=ctk.CTkFont(family=self.safe_font, size=12)
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
        """Mostrar la imagen en la interfaz"""
        try:
            # Cargar imagen con PIL
            pil_image = Image.open(image_path)
            self.current_image = pil_image.copy()
            
            # Obtener información de la imagen
            width, height = pil_image.size
            file_size = os.path.getsize(image_path) / 1024  # KB
            
            self.info_label.configure(
                text=f"Dimensiones: {width}x{height} | Tamaño: {file_size:.1f} KB | Red neuronal: 700x600"
            )
            
            # Redimensionar para mostrar (manteniendo aspecto) con mejor calidad
            display_size = (300, 250)
            # Usar LANCZOS para mejor calidad de redimensionamiento
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convertir a formato compatible con CustomTkinter usando mejor escalado
            photo = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(pil_image.width, pil_image.height)  # Usar tamaño exacto
            )
            
            # Actualizar el label con la imagen
            self.image_label.configure(image=photo, text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar la imagen: {str(e)}")
    
    def process_image(self):
        """Procesar la imagen con la red neuronal"""
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
        
        # Redimensionar imagen a 700x600 para la red neuronal
        resized_image = self.resize_for_neural_network(self.current_image)
        
        # Preprocesar imagen para tu red neuronal
        # Convertir PIL a numpy array
        image_array = np.array(resized_image)
        
        # Convertir a RGB si es necesario
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Normalizar (común en redes neuronales)
        # processed_image = image_array / 255.0
        
        # AQUÍ LLAMAS A TU RED NEURONAL:
        # prediction = tu_modelo.predict(processed_image)
        # confidence = float(prediction[0]) * 100
        # is_recognized = prediction[0] > 0.5  # umbral de decisión
        
        # SIMULACIÓN (reemplazar con tu código real):
        import random
        confidence = random.uniform(60, 95)
        is_recognized = confidence > 75
        
        return is_recognized, confidence
    
    def resize_for_neural_network(self, pil_image):
        """
        Redimensiona la imagen a 700x600 píxeles para la red neuronal
        
        Args:
            pil_image: Imagen PIL original
            
        Returns:
            PIL Image redimensionada a 700x600
        """
        try:
            # Crear una copia de la imagen original
            image_copy = pil_image.copy()
            
            # Método 1: Redimensionamiento directo (puede distorsionar)
            # resized_image = image_copy.resize((700, 600), Image.Resampling.LANCZOS)
            
            # Método 2: Redimensionamiento manteniendo aspecto + relleno (recomendado)
            resized_image = self.resize_with_padding(image_copy, (700, 600))
            
            return resized_image
            
        except Exception as e:
            print(f"Error al redimensionar imagen: {e}")
            return pil_image
    
    def resize_with_padding(self, image, target_size):
        """
        Redimensiona la imagen manteniendo el aspecto y agregando padding negro
        
        Args:
            image: Imagen PIL
            target_size: tuple (width, height) - tamaño objetivo
            
        Returns:
            PIL Image redimensionada con padding
        """
        target_width, target_height = target_size
        
        # Calcular el ratio de redimensionamiento
        img_width, img_height = image.size
        ratio = min(target_width / img_width, target_height / img_height)
        
        # Calcular nuevas dimensiones
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Redimensionar imagen
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crear imagen con fondo negro del tamaño objetivo
        final_image = Image.new('RGB', target_size, (0, 0, 0))
        
        # Calcular posición para centrar la imagen
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Pegar la imagen redimensionada en el centro
        final_image.paste(resized_img, (x_offset, y_offset))
        
        return final_image
    
    def resize_direct(self, image, target_size):
        """
        Redimensionamiento directo (puede distorsionar la imagen)
        
        Args:
            image: Imagen PIL
            target_size: tuple (width, height)
            
        Returns:
            PIL Image redimensionada directamente
        """
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def resize_crop_center(self, image, target_size):
        """
        Redimensiona y recorta desde el centro
        
        Args:
            image: Imagen PIL
            target_size: tuple (width, height)
            
        Returns:
            PIL Image redimensionada y recortada
        """
        target_width, target_height = target_size
        img_width, img_height = image.size
        
        # Calcular ratio para cubrir completamente el área objetivo
        ratio = max(target_width / img_width, target_height / img_height)
        
        # Redimensionar
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calcular área de recorte desde el centro
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        # Recortar
        cropped_img = resized_img.crop((left, top, right, bottom))
        
        return cropped_img
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

# Función principal
def main():
    app = FaceRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()