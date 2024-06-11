from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils
module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(
    nestml_neuron_model="controller_module.nestml",
    module_name="controller_module")
