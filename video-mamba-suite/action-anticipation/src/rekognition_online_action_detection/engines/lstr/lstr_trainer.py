# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..base_trainers.perframe_det_trainer import do_perframe_det_train


from ..engines import TRAINERS as registry
@registry.register('LSTR')
def do_lstr_train(cfg,
                  data_loaders,
                  model,
                  criterion,
                  optimizer,
                  scheduler,
                  device,
                  checkpointer,
                  logger):
    do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          device,
                          checkpointer,
                          logger)
