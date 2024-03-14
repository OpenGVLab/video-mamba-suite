# DATASET

## THUMOS14
```bash
wget https://utexas.box.com/shared/static/7jr33g7mtoowsrzn99vecebu9co4wywv.zip -O target_perframe.zip
wget https://utexas.box.com/shared/static/fbetd0331iod7jx7udfbckn9359mrp6o.zip -O rgb_kinetics_resnet50.zip
wget https://utexas.box.com/shared/static/kdzeeztwlaphe8zcun5ebavv2pd37fxb.zip -O flow_kinetics_bninception.zip
wget https://utexas.box.com/shared/static/8tneyw7npy7gsgzydlu3610czlzmhs4k.zip -O flow_nv_kinetics_bninception.zip
unzip target_perframe.zip -d target_perframe/ && rm target_perframe.zip
unzip rgb_kinetics_resnet50.zip -d rgb_kinetics_resnet50/ && rm rgb_kinetics_bninception.zip
unzip flow_kinetics_bninception.zip -d flow_kinetics_bninception/ && rm flow_kinetics_bninception.zip
unzip flow_nv_kinetics_bninception.zip -d flow_nv_kinetics_bninception/ && rm flow_nv_kinetics_bninception.zip
```

## THUMOS14 (ANet-1.3)
```bash
wget https://utexas.box.com/shared/static/avtdkeegkh5kl7ajg4ltqhd3ai33bb8m.zip -O rgb_anet_resnet50.zip
wget https://utexas.box.com/shared/static/rhvihb33e54ro07zsmcbgku16cikk2g4.zip -O flow_anet_resnet50.zip
unzip rgb_anet_resnet50.zip -d rgb_anet_resnet50/ && rm rgb_anet_resnet50.zip
unzip flow_anet_resnet50.zip -d flow_anet_resnet50/ && rm flow_anet_resnet50.zip
```

## EK100
```bash
wget https://utexas.box.com/shared/static/kypifujsplkg0ud7q955amgvoxflqzx5.zip -O rgb_kinetics_bninception.zip
wget https://utexas.box.com/shared/static/2aga6r29o4zdziog3y89aliauguiqhmn.zip -O flow_kinetics_bninception.zip
wget https://utexas.box.com/shared/static/xi1xowkhlmi079suwwq6dlez44lb846e.zip -O target_perframe.zip
wget https://utexas.box.com/shared/static/e9yes31rblmuzb5mdrf3gy1mb7af7a63.zip -O verb_perframe.zip
wget https://utexas.box.com/shared/static/vmg478wjbcf83wc0adw0t9yxduxjqna9.zip -O noun_perframe.zip
unzip rgb_kinetics_bninception.zip -d rgb_kinetics_bninception/ && rm rgb_kinetics_bninception.zip
unzip flow_kinetics_bninception.zip -d flow_kinetics_bninception/ && rm flow_kinetics_bninception.zip
unzip target_perframe.zip -d target_perframe/ && rm target_perframe.zip
unzip verb_perframe.zip -d verb_perframe/ && rm verb_perframe.zip
unzip noun_perframe.zip -d noun_perframe/ && rm noun_perframe.zip
```
