# from IPython.display import clear_output, Image, display, HTML
# %matplotlib notebook
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import base64
import json
from PIL import Image, ImageFont, ImageDraw
import pdb
import argparse
import os
from tqdm import tqdm


def get_frame_caption(frame_time, dense_captions, ranking=False):
    frame_captions = []
    idx_list = []
    for event in dense_captions:
        s, e = event['timestamp']
        if frame_time >= s and frame_time <= e:
            frame_captions.append(event)
            idx_list.append(event['original_id'])
    temperature = 1
    if ranking:
        scorer = lambda p: p['sentence_score'] / (float(len(p['sentence'].split())) ** (temperature) + 1e-5) + \
                       1.0 * p['proposal_score'] * (1 - np.abs(frame_time - 0.5 * (p['timestamp'][0] + p['timestamp'][1])) / (
                                   p['timestamp'][1] - p['timestamp'][0] + 1e-8))

        frame_captions = sorted(frame_captions, key=scorer, reverse=True)
    else:
        frame_captions = sorted(frame_captions, key=lambda p: p['timestamp'])
    return frame_captions, idx_list


def paint_text(im, chinese, font, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    if not isinstance(chinese, str):
        chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def processImg(img, cur_time, title, dense_captions, prop_idx, n_caption=3, output_language='en'):
    scale = 1.0
    basic_text_height = 50
    text_height = int(basic_text_height * scale)
    font_size = int(text_height * 0.8)

    h, w, c = img.shape
    last_time = cur_time
    cur_time = time.time()
    img_fps = 1. / (cur_time - last_time + 1e-8)
    bg_img = np.zeros_like(img)
    cv2.rectangle(bg_img, (0, 0), (len(title) * text_height // 2, text_height), (120, 120, 120), -1, 1, 0)
    cv2.rectangle(bg_img, (0, h - text_height * n_caption), (w, h), (120, 120, 120), -1, 1, 0)
    mask = bg_img / 255.
    alpha = 0.5
    img = img * (mask == 0) + alpha * img * (mask > 0) + (1 - alpha) * mask
    img = img.astype('uint8')
    if output_language == 'zh-cn':
        font = ImageFont.truetype('visualization/NotoSansCJK-Bold.otf', font_size)
    elif output_language == 'en':
        font = ImageFont.truetype("visualization/Arial.ttf", font_size)
    else:
        font = ImageFont.truetype("/path/to/your.font.ttf", font_size)
    img = paint_text(img, title, font, (10, 0), color=(255, 255, 255))
    for i, (proposal) in enumerate(dense_captions):
        caption, timestamp = proposal['sentence'], proposal['timestamp']
        caption = '{:2.1f}s-{:2.1f}s: {}'.format(timestamp[0], timestamp[1], caption)
        ptText = (10, h - text_height * n_caption + i * text_height)
        if i in prop_idx:
            img = paint_text(img, caption, font, ptText, color=(255, 0, 0))
        else:
            img = paint_text(img, caption, font, ptText, color=(255, 255, 255))

    return img, cur_time, img_fps

def vid_show(vid_path, captions, save_mp4, save_mp4_path, output_language='en'):
    start_time = time.time()
    cur_time = time.time()
    video = cv2.VideoCapture(vid_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    print('fps: {}, duration: {}, frames: {}'.format(fps, duration, frame_count))
    img_fps = fps
    n = 0
    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWriter = cv2.VideoWriter(save_mp4_path, fourcc, fps, (1280, 720))

    if not output_language == 'en':
        for proposal in captions:
            caption = translator.translate(proposal['sentence'], lang_src='en', lang_tgt=output_language)
            proposal['sentence'] = caption
    for i, proposal in enumerate(captions):
        proposal['original_id'] = i
    captions = sorted(captions, key=lambda p: p['timestamp'])

    for frame_id in tqdm(range(int(frame_count))):
        ret, frame = video.read()
        if n >= int(fps / img_fps) or save_mp4:
            n = 0
            # clear_output(wait=True)
        else:
            n += 1
            continue
        if not ret:
            break
        lines, columns, _ = frame.shape
        frame = cv2.resize(frame, (1280, 720))
        frame_time = frame_id / fps
        if opt.show_all_caption_per_frame:
            frame_captions, highlight_idx = get_frame_caption(frame_time, captions, ranking=False)
            captions_to_show = captions
            n_caption = len(captions)
        else:
            frame_captions, highlight_idx = get_frame_caption(frame_time, captions, ranking=True)
            captions_to_show = frame_captions
            n_caption = min(3, len(captions_to_show))

        title = '{:.1f}s/{:.1f}s'.format(frame_time, duration)
        frame, cur_time, img_fps = processImg(frame, cur_time, title, captions_to_show, highlight_idx, output_language=output_language, n_caption=n_caption)
        if not save_mp4:
            plt.axis('off')
            plt.imshow(frame[:, :, ::-1])
            plt.show()
        # control fps
        if save_mp4:
            videoWriter.write(frame)

    if save_mp4:
        videoWriter.release()
        print('output videos saved at {}, process time: {} s'.format(save_mp4_path, cur_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_language', type=str, default='en',
                        help='refer to /path/to/miniconda3/envs/PDVC/lib/python3.7/site-packages/google_trans_new/constant.py for more information')
    parser.add_argument('--output_mp4_folder', type=str, default=None)
    parser.add_argument('--input_mp4_folder', type=str, required=True)
    parser.add_argument('--dvc_file', type=str, required=True)
    parser.add_argument('--show_all_caption_per_frame', type=int, default=False)
    opt = parser.parse_args()
    if not opt.output_language == 'en':
        from google_trans_new import google_translator
        translator = google_translator()
    d = json.load(open(opt.dvc_file))['results']
    for vid, dense_captions in d.items():
        if opt.output_mp4_folder is None:
            opt.output_mp4_folder = opt.input_mp4_folder + '_output'
        if not os.path.exists(opt.output_mp4_folder):
            os.mkdir(opt.output_mp4_folder)
        output_mp4_path = os.path.join(opt.output_mp4_folder, vid + '.mp4')

        input_mp4_path = os.path.join(opt.input_mp4_folder, vid + '.mp4')
        print('process video: {} --> output: {}'.format(input_mp4_path, output_mp4_path))
        if not os.path.exists(input_mp4_path):
            print('vidoe {} does not exists, skip it.')
            continue
        vid_show(input_mp4_path, dense_captions, save_mp4=True, save_mp4_path=output_mp4_path,
                 output_language=opt.output_language)
