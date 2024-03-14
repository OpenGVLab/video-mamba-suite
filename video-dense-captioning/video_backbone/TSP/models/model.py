import torch
from torch import nn
from .backbone import r2plus1d_34, r2plus1d_18, r3d_18


class Model(nn.Module):

    def __init__(self, backbone, num_classes, num_heads=1, concat_gvf=False, progress=True, **kwargs):
        '''
        Args:
            backbone (string): The name of the backbone architecture. Supported architectures: r2plus1d_34, r2plus1d_18, and r3d_18.
            num_heads (int): The number of output heads
            num_classes (list of int): The number of labels per head
            concat_gvf (bool): If True and num_heads == 2, then concat global video features (GVF) to clip
                features before applying the second head FC layer.
            progress (bool): If True, displays a progress bar of the download to stderr
            **kwargs: keyword arguments to pass to backbone architecture constructor
        '''
        super().__init__()
        print(f'<Model>: backbone {backbone} num_classes {num_classes} num_heads {num_heads} kwargs {kwargs}')
        assert len(num_classes) == num_heads, f'<Model>: incompatible configuration. len(num_classes) must be equal to num_heads'
        assert num_heads == 1 or num_heads == 2, f'<Model>: num_heads = {num_heads} must be either 1 or 2'

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.concat_gvf = concat_gvf

        self.features, self.feature_size = Model._build_feature_backbone(backbone, progress, **kwargs)

        if self.num_heads == 1:
            self.fc = Model._build_fc(self.feature_size, num_classes[0])
        else:
            self.fc1 = Model._build_fc(self.feature_size, num_classes[0])
            self.fc2 = Model._build_fc(2 * self.feature_size if self.concat_gvf else self.feature_size, num_classes[1])

    def forward(self, x, gvf=None, return_features=False):
        features = self.features(x)
        if self.num_heads == 1:
            logits = [self.fc(features)]
        else:
            logits = [self.fc1(features)]
            if self.concat_gvf:
                assert gvf is not None, 'Forward pass expects a global video feature input but got None'
                logits.append(self.fc2(torch.cat([features, gvf], dim=-1)))
            else:
                logits.append(self.fc2(features))

        return (logits, features) if return_features else logits

    @staticmethod
    def _build_feature_backbone(backbone, progress, **kwargs):
        if backbone == 'r2plus1d_34': builder = r2plus1d_34
        elif backbone == 'r2plus1d_18': builder = r2plus1d_18
        elif backbone == 'r3d_18': builder = r3d_18
        else:
            raise ValueError(f'<Model>: {backbone} is an invalid architecture type. '
                f'Supported  architectures: r2plus1d_34, r2plus1d_18, and r3d_18')

        feature_backbone = builder(pretrained=True, progress=progress, **kwargs)

        # remove the FC layer of the backbone
        feature_size = feature_backbone.fc.in_features
        feature_backbone.fc = nn.Sequential()

        return feature_backbone, feature_size

    @staticmethod
    def _build_fc(in_features, out_features):
        fc = nn.Linear(in_features, out_features)
        nn.init.normal_(fc.weight, 0, 0.01)
        nn.init.constant_(fc.bias, 0)
        return fc
