from .LSTM import LightCaptioner
from .Puppet import PuppetCaptionModel
from .LSTM_DSA import LSTMDSACaptioner

def build_captioner(opt):
    if opt.caption_decoder_type == 'none':
        caption_embed = PuppetCaptionModel(opt)

    elif opt.caption_decoder_type == 'light':
        opt.event_context_dim = None
        opt.clip_context_dim = opt.hidden_dim
        caption_embed = LightCaptioner(opt)

    elif opt.caption_decoder_type == 'standard':
        opt.event_context_dim = None
        opt.clip_context_dim = opt.hidden_dim
        caption_embed = LSTMDSACaptioner(opt)

    else:
        raise ValueError('caption decoder type is invalid')
    return caption_embed

