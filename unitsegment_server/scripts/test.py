from __future__ import print_function

import time
from modules.cnnvisualizer import single_unitsegment as su

start_time = time.time()
visualizer = su.Cnnvisualizer()
visualizer.load_model()
end_time = time.time()
print('[time] loading model: %0.2f' % (end_time - start_time))

for index in range(3560, 3570):
    for layer in range(1, 5):
        for unit in range(10):
            start_time = time.time()

            visualizer.set_layer_unit('layer%d' % layer, unit)

            input_path = 'images/received/00%d.jpg' % index 
            input_image = visualizer.read_input_image(input_path)

            segment = visualizer.generate_unitsegment(input_image)
            output_path = visualizer.build_output_path('images/results', '00%d' % index)
            visualizer.save_unitsegment(output_path, segment)

            end_time = time.time()
            print('[time] generate one unitsegment: %0.2f' % (end_time - start_time))
            print('success save unitsegment to', output_path)

print('finished..')

