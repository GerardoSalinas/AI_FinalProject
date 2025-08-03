from PIL import Image
import os
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass
    
    def resize_black_and_white(self, pil_image):
        """
        Redimensiona la imagen a 700x600 pixeles y lo pasa a blanco y negro
        
        Args:
            pil_image: Imagen PIL original
            
        Returns:
            PIL Image redimensionada a 700x600
        """
        try:
            image_copy = pil_image.copy()
            resized_image = image_copy.resize((700, 600), Image.Resampling.LANCZOS)
            grayscale_image = resized_image.convert("L")
            # resized_image.show();
            return grayscale_image
        except Exception as e:
            print(f"Error al redimensionar imagen: {e}")
            return pil_image
    
    def image_to_vector(self,pil_image):
        img_array = np.array(pil_image) / 255.0
        return img_array.flatten() 
    
    def map_matrix_values(self,pixel_matrix):
        """
        Mapea los valores de los pixeles, 255 -> 1 y 0 -> 0
        
        Args:
            pixel_matrix: Matriz de pixeles en blanco y negro
            
        Returns:
            zero_one_pixel_matrix: Matriz de pixeles con solo ceros y unos
        """
        zero_one_pixel_matrix = (pixel_matrix == 255).astype(np.uint8)
        return zero_one_pixel_matrix
    
    def batch_processing(self,source_directory,destination_directory):
        content = os.listdir(source_directory)
        counter = 0
        dataset = []
        for file in content:
            initial_image_path = source_directory + file
            newImage = Image.open(initial_image_path)
            resized_image = self.resize_black_and_white(newImage)
            new_image_path = destination_directory + file
            pixel_vector = self.image_to_vector(resized_image)
            # pixel_vector = self.map_matrix_values(pixel_vector)
            tag = 0
            if ("gerardo" in file):
                tag = 1
            dataset.append((pixel_vector,tag))
            resized_image.save(new_image_path)
            counter += 1
        print(f"Se insertaron {counter} imagenes")
        return dataset

def quick_test():
    print("Prueba rápida de ImageProcessor")
    
    test_image = Image.open('images/objetivo/2.jpg')
    print(f"Dimensiones originales: {test_image.size}")
    
    processor = ImageProcessor()
    my_image_dataset = processor.batch_processing('/home/gasallinas/Documents/Clases/IA/finalProject/images/objetivo/','/home/gasallinas/Documents/Clases/IA/finalProject/images/black_white_images/',1)
    # other_image_dataset = processor.batch_processing('/home/gasallinas/Documents/Clases/IA/finalProject/images/otro/','/home/gasallinas/Documents/Clases/IA/finalProject/images/black_white_images/',0)
    # full_image_dataset = []
    # full_image_dataset.append(my_image_dataset)
    # full_image_dataset.append(other_image_dataset)
    # result = processor.resize_black_and_white(test_image)
    # image_pixel_matrix = processor.image_to_list(result)
    # pixel_matrix = processor.map_matrix_values(image_pixel_matrix)
    # print(pixel_matrix)
    
    # if result:
    #     print(f"Dimensiones actuales: {result.size}")
    #     result.save("/home/gasallinas/Documents/Clases/IA/finalProject/images/black_white_images/test_resized_grayscale.jpg")
    #     print("Imagen guardada como 'test_resized.jpg'")
    # else:
    #     print("Error: No se pudo redimensionar la imagen")

if __name__ == "__main__":
    quick_test()