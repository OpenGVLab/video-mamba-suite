ANet-Entities val/test splits (re-split from ANet-caption val_1 and val_2 splits):
https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz

ANet-caption original splits:
http://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip

Experiment settings:
Training: use GT segments/sentences in `train.json`, 
Validation: use GT segments in `anet_entities_val_1.json`, evaluate against references `anet_entities_val_1_para.json` and `anet_entities_val_2_para.json`
Test: use GT segments in `anet_entities_test_1.json`, evaluate against references `anet_entities_test_1_para.json` and `anet_entities_test_2_para.json`



