import nest
from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(
    nestml_neuron_model="controller_module.nestml", module_name="controller_module"
)

# import os
# curr_dir = os.getcwd()
# module_dir = os.path.join(curr_dir,'target')
# print(module_dir)
# os.environ['NEST_MODULE_PATH'] = module_dir
