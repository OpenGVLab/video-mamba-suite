import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
 

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content
 
 
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends
 
 
def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score
 
 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
 
def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.get_cmap('seismic')
    # color_map =
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=20)
 
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def segment_bars_with_confidence(save_path, confidence, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('seismic')
 
    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0)
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    interval = 1 / (num_pics+1)
    for i, label in enumerate(labels):
        i = i + 1
        ax1 = fig.add_axes([0, 1-i*interval, 1, interval])
        ax1.imshow([label], **barprops)
 
    ax4 = fig.add_axes([0, interval, 1, interval])
    ax4.set_xlim(0, len(confidence))
    ax4.set_ylim(0, 1)
    ax4.plot(range(len(confidence)), confidence)
    ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def func_eval(dataset, recog_path, file_list):
    ground_truth_path = "./data/" + dataset + "/groundTruth/"
    mapping_file = "./data/" + dataset + "/mapping.txt"
    list_of_videos = read_file(file_list).split('\n')[:-1]
 
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
 
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

 
    for vid in list_of_videos:
 
         
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
 
        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()
 

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(list_of_videos)
#     print("Acc: %.4f" % (acc))
#     print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
#         print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1
 
    return acc, edit, f1s

def main():
    cnt_split_dict = {
        '50salads':5,
        'gtea':4,
        'breakfast':4
    }
    
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--result_dir', default='results')
    parser.add_argument('--mamba', action='store_true')
    parser.add_argument('--drop_path_rate', type=float, default=0.3)
    parser.add_argument('--channel_mask_rate', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=120)

    args = parser.parse_args()
    if args.mamba:
        addstr = '_mamba_dp%.2f_l%d_m%.2f_e%d' % (args.drop_path_rate, args.num_layers, args.channel_mask_rate, args.num_epochs)
    else:
        addstr = ''

    acc_all = 0.
    edit_all = 0.
    f1s_all = [0.,0.,0.]
    
    if args.split == 0:
        for split in range(1, cnt_split_dict[args.dataset] + 1):
            recog_path = "./{}/".format(args.result_dir)+args.dataset+"/split_{}".format(split)+addstr+"/"
            file_list = "./data/"+args.dataset+"/splits/test.split{}".format(split)+".bundle"
            acc, edit, f1s = func_eval(args.dataset, recog_path, file_list)
            acc_all += acc
            edit_all += edit
            f1s_all[0] += f1s[0]
            f1s_all[1] += f1s[1]
            f1s_all[2] += f1s[2]
        
        acc_all /=  cnt_split_dict[args.dataset]
        edit_all /= cnt_split_dict[args.dataset]
        f1s_all = [i / cnt_split_dict[args.dataset] for i in f1s_all]
    else:
        split = args.split
        recog_path = "./{}/".format(args.result_dir)+args.dataset+"/split_{}".format(split)+addstr+"/"
        file_list = "./data/"+args.dataset+"/splits/test.split{}".format(split)+".bundle"
        acc_all, edit_all, f1s_all = func_eval(args.dataset, recog_path, file_list)
    
    print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)


if __name__ == '__main__':
    main()