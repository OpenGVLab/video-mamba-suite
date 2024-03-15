# coding:utf-8
import json

# file_path_list = ["data/captiondata/train_modified.json", "data/captiondata/val_1.json", "data/captiondata/val_2.json"]
file_path_list = ["data/captiondata/yc2/yc2_train.json", "data/captiondata/yc2/yc2_val.json"]

count_threshold = 2 # 4 for anet, 2 for youcook2
# output_path = './data/vocabulary_activitynet.json'
output_path = './data/vocabulary_youcook2.json'

mark = [',', ':', '!', '_', ';', '-', '.', '?', '/', '"', '\\n', '\\']

count_vocal = {}

for file_path in file_path_list:
    data = json.load(open(file_path))
    video_ids = data.keys()
    print('video num of ' + file_path.split('/')[-1], len(video_ids))
    for video_id in video_ids:
        sentences = data[video_id]["sentences"]
        for sentence in sentences:
            for m in mark:
                if m in sentence:
                    sentence = sentence.replace(m, " ")
                sentence = sentence.replace("  ", " ")
                sentence = sentence.replace("  ", " ")
                sentence = sentence.replace("  ", " ")

            sentence = sentence.lstrip()
            sentence = sentence.rstrip()
            sentence = sentence.lower()
            sentence = sentence.split(" ")
            length = len(sentence)

            # print(sentence)
            for word in sentence:
                # print(type(word))
                for m in word:
                    if m == ' ':
                        print('warning !')
                        word = word.replace(m, '')
                if word == '':
                    print('warning !')
                    pass
                count_vocal[word] = count_vocal.get(word, 0) + 1

print("total word:", sum(count_vocal.values()))
count_vocal['<bos>'] = 1e10
count_vocal['<eos>'] = 1e10
vocab = [word for word, n in count_vocal.items() if n >= count_threshold]
bad_word = [word for word, n in count_vocal.items() if n < count_threshold]
bad_count = sum(count_vocal[word] for word in bad_word)

vocab.append('UNK')
print("number of vocab:", len(vocab))
print("number of bad word:", len(bad_word))
print("number of unks:", bad_count)

itow = {i + 1: w for i, w in enumerate(vocab)}
wtoi = {w: i + 1 for i, w in enumerate(vocab)}
print(len(itow))
print(len(wtoi))

json.dump({'ix_to_word': itow,
           'word_to_ix': wtoi}, open(output_path, 'w'))
print("saving vocabulary file to {}".format(output_path))