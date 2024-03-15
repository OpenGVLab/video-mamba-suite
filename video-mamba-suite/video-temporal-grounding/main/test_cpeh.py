s3_path = 's3://videos/univtg_feature/QVHighlights/txt_clip'
import mmengine
import os
import io
data = mmengine.get(os.path.join(s3_path,'0.npz'))
import numpy as np
q_feat = io.BytesIO(data)
q_feat = np.load(q_feat)['last_hidden_state'].astype(np.float32)
print(q_feat.shape)
print(q_feat[0])