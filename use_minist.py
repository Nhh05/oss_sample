import os
import numpy as np

'''
from PNGProcessor import PNGProcessor
test_png_path = "1.png"
png_processor = PNGProcessor()
pixels = png_processor.open_png(test_png_path)
'''
#그림판 이미지는 RGB 이기때문에 PIL 라이브러리 사용
from PIL import Image
test_png_path = "2.png"
img = Image.open(test_png_path).convert("L")
img = img.resize((28, 28), Image.Resampling.LANCZOS)
pixels = np.array(img).tolist()

from PNGProcessor import PNGProcessor
png_processor = PNGProcessor()
#pixels = png_processor.invert_image_colors(pixels)
for y in pixels:
    for j in y:
        if j<10:
            print("  ",j,end="")
        elif j<100:
            print(" ",j,end="")
        else:
            print("",j,end="")
    print()

from DataPreProcessor import DataPreProcessor 
data_processor = DataPreProcessor()
flattened_data = data_processor.flatten(pixels)
scaled_data = data_processor.scale_data([flattened_data])


from MLPForMINIST import MLPForMINIST 
save_folder = "mlp_model_params"
mlp_model = MLPForMINIST(28 * 28, 128 ,10,0.2)
mlp_model.load_model(save_folder)
predict_data = mlp_model.predict([flattened_data])


print("predict_data: ",predict_data)