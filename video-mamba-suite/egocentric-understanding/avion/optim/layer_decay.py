class LayerDecayValueAssigner:
    def __init__(self, values):
        self.values = values
        self.num_layers = len(self.values)

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if var_name in ['cls_token', 'mask_token', 'pos_embed']:
            return 0
        elif var_name.startswith('patch_embed'):
            return 0
        elif var_name.startswith('rel_pos_bias'):
            return self.num_layers - 1
        elif var_name.startswith('blocks') or var_name.startswith('layers'):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return self.num_layers - 1
