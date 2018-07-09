import os
from modules.cnnvisualizer import single_unitsegment as su

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Cnnvisualizer(object):
    # 构造函数里加载模型，比如 tensorflow 的 graph, sess 等
    def __init__(self):
        self.visualizer = su.Cnnvisualizer()
        self.visualizer.load_model()

    # 需要对外提供一个 API，可以直接拿到你们的结果
    def generate_unitsegment(self, image_path, layer, unit):
        self.visualizer.set_layer_unit(layer, unit)
        input_image = self.visualizer.read_input_image(image_path)
        segment = self.visualizer.generate_unitsegment(input_image)

        filename, ext = os.path.basename(image_path).split('.')
        output_path = self.visualizer.build_output_path("static/outputs", filename)
        self.visualizer.save_unitsegment(output_path, segment)

        output_basename = os.path.basename(output_path)
        return output_basename
    

print("生成 Cnnvisualizer 实例......")
cnnvisualizer_instance = Cnnvisualizer()
print("Cnnvisualizer 实例生成完毕......")
