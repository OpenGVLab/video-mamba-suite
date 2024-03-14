# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        if module is not None:
            _register_generic(self, module_name, module)
            return

        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
