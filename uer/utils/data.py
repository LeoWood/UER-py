# -*- encoding:utf-8 -*-
import os
import torch
import codecs
import random
import pickle
import re
from tqdm import tqdm
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed

# import pkuseg
# pku_seg = pkuseg.pkuseg(model_name="medicine",user_dict="../uer/utils/pku_seg_dict.txt")
# pku_seg_pos = pkuseg.pkuseg(model_name="medicine",user_dict="../uer/utils/pku_seg_dict.txt",postag=True)


# pos_dict = {}
# with open('../uer/utils/pos_tags_old.txt','r',encoding='utf-8') as f:
#     i = 0
#     for line in f.readlines():
#         if line:
#             pos_dict[line.strip().split()[0]] = i
#             i += 1


# # 获取本地术语表
# a = []
# term_dict = {}
# with open('../uer/utils/medical_terms/medical_terms(final).txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         line = line.strip()
#         a.append(line)
#         term_dict[line.lower()] = 1

# term_set = set(a)

# max_num = max([len(line) for line in a])


def mask_seq(src, vocab_size):
    """
    mask input sequence for MLM task
    args:
        src: a list of tokens
        vocab_size: the vocabulary size
    """
    tgt_mlm = []
    for (i, token) in enumerate(src):
        if token == CLS_ID or token == SEP_ID:
            continue
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.8:
                src[i] = MASK_ID
            elif prob < 0.9:
                while True:
                    rdi = random.randint(1, vocab_size-1)
                    if rdi not in [CLS_ID, SEP_ID, MASK_ID]:
                        break
                src[i] = rdi
            tgt_mlm.append((i, token))
    return src, tgt_mlm


def merge_dataset(dataset_path, workers_num):
        # Merge datasets.
        f_writer = open(dataset_path, "wb")
        for i in range(workers_num):
            tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
            while True:
                tmp_data = tmp_dataset_reader.read(2^20)
                if tmp_data:
                    f_writer.write(tmp_data)
                else:
                    break
            tmp_dataset_reader.close()
            os.remove("dataset-tmp-"+str(i)+".pt")
        f_writer.close()


def seg_char(sent):
    """
    把句子按字分开，不破坏英文及数字结构
    """
    # 按中文汉字分割
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    parts = pattern.split(sent)
    parts = [w for w in parts if len(w.strip())>0]

    # 按英文标点符号分割
    chars_list = []
    pattern = re.compile(r'([,?:;\'"!()<>，。；’‘、！])')
    for part in parts:
        chars = pattern.split(part)
        chars = [w for w in chars if len(w.strip())>0]
        chars_list += chars

    return chars_list

def max_match(txt, ano_dict, max_num):
    word_list = seg_char(txt) # 中文单字切割，保留英文和数字
    # print(word_list)
    new_word_list = []
    term_labels = []
    N = len(word_list)
    k = max_num
    i = 0
    while i < N:
        if i <= N - k:
            j = k
            while j > 0:
                token_tmp = ''.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp,'！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    term_labels.append(1)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                new_word_list.append(word_list[i])
                term_labels.append(0)
                i += 1
        else:
            j = N - i
            while j > 0:
                token_tmp = ''.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp, '！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    term_labels.append(1)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                new_word_list.append(word_list[i])
                term_labels.append(0)
                i += 1
    return new_word_list,term_labels

# ## 最大匹配分词
# txt = '我爱北京天安门，天安门上太阳升'
#
# ano_dict={'天安门':1,'北京':1,'太阳':1}
#
# max_num = 2
#
# print(max_match(txt,ano_dict,max_num))
# exit()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.stats_tokens = args.stats_tokens
        self.stats_instances = args.stats_instances
        self.stats_lines = args.stats_lines


    def build_and_save(self, workers_num):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """

        if self.stats_instances:
            ins_count = 0
            with open(self.dataset_path,'rb') as f:
                i = 0
                while True:
                    try:
                        instance = pickle.load(f)
                        ins_count += len(instance)
                    except EOFError:
                        break
                    i += 1
            print("instances: ", ins_count)
            exit()

        if self.stats_tokens:
            tokens_count = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                for line in tqdm(f.readlines()):
                    line = line.strip()
                    if line:
                        tokens_count += len(self.tokenizer.tokenize(line))
            print("tokens: ",tokens_count)
            exit()
        
        if self.stats_lines:
            lines_count = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                for line in tqdm(f.readlines()):
                    line = line.strip()
                    if line:
                        lines_count += 1
            print("lines: ",lines_count)
            exit()


        lines_num = count_lines(self.corpus_path)
        print("Total %d lines in courpus ... " % lines_num)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert(workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i+1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.f_read = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.add_pos = args.add_pos

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.f_read)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.f_read.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.f_read.close()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences, 
    and each sentence occupies a single line. 
    Documents in corpus must be separated by empty lines.
    """
    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1
                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        # Build instances from documents.                    
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, f_write)
                        # Clear buffer.
                        docs_buffer = []
                        instances = []
                    continue
                sentence = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                if len(sentence) > 0:
                    document.append(sentence)
        
                if pos >= end - 1:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, f_write)
                    break
        f_write.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_random_next = 0

                    # Random next
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    # Actual next
                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    # assert len(tokens_a) >= 1
                    # assert len(tokens_b) >= 1

                    src = []

                    src.append(CLS_ID)
                    for token in tokens_a:
                        src.append(token)

                    src.append(SEP_ID)

                    seg_pos = [len(src)]

                    for token in tokens_b:
                        src.append(token)
            
                    src.append(SEP_ID)

                    seg_pos.append(len(src))

                    src, tgt_mlm = mask_seq(src, len(self.vocab))
                    
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    instance = (src, tgt_mlm, is_random_next, seg_pos)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """ truncate sequence pair to specific length """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
                
            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            # assert len(trunc_tokens) >= 1

            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[1])
            if masked_words_num == 0:
                continue
            
            for ins in instances:
                src.append(ins[0])
                tgt_mlm.append([0]*len(ins[0]))
                for mask in ins[1]:
                    tgt_mlm[-1][mask[0]] = mask[1]
                is_next.append(ins[2])
                seg.append([1]*ins[3][0] + [2]*(ins[3][1]-ins[3][0]) + [PAD_ID]*(len(ins[0])-ins[3][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class LmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [SEP_ID]
                tgt_backward = [CLS_ID] + src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt_forward = tgt_forward[:self.seq_length]
                    tgt_backward = tgt_backward[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt_forward.append(PAD_ID)
                        tgt_backward.append(PAD_ID)
                        seg.append(PAD_ID)
                
                pickle.dump((src, tgt_forward, tgt_backward, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class BilmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_forward.append(ins[1])
                tgt_backward.append(ins[2])
                seg.append(ins[3])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = " ".join(line[1:])
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [CLS_ID] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                elif len(line) == 3: # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [CLS_ID] + tokens_a + [SEP_ID]
                    src_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                    src_b = tokens_b + [SEP_ID]

                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                else:
                    pass

                if pos >= end - 1:
                    break

        f_write.close()


class ClsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.dup_factor = args.dup_factor

    
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    try:
                        f.readline()
                    except:
                        continue
                    finally:
                        pos += 1
                pbar = tqdm(total=end - start)
                pbar.set_description("Worker %d:" % proc_id)
                while True:
                    pbar.update(1)
                    try:
                        line = f.readline()
                    except:
                        continue
                    finally:
                        pos += 1
                    
                    line = line.strip()
                    if not line:
                        continue

                    src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]

                    if len(src) > self.seq_length:
                        src = src[:self.seq_length]
                    seg = [1] * len(src)

                    src, tgt = mask_seq(src, len(self.vocab))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        seg.append(PAD_ID)

                    pickle.dump((src, tgt, seg), f_write)

                    if pos >= end - 1:
                        break
                pbar.close()
        f_write.close()


class MlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[1])
            if masked_words_num == 0:
                continue

            for ins in instances:
                src.append(ins[0])
                seg.append(ins[2])
                tgt.append([0]*len(ins[0]))
                for mask in ins[1]:
                    tgt[-1][mask[0]] = mask[1]

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class Csci_mlmDataset(Dataset):
    
    def __init__(self, args, vocab, tokenizer):
        super(Csci_mlmDataset, self).__init__(args, vocab, tokenizer)
        self.dup_factor = args.dup_factor

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    try:
                        f.readline()
                    except:
                        continue
                    finally:
                        pos += 1
                pbar = tqdm(total=end - start)
                pbar.set_description("Worker %d:" % proc_id)
                while True:
                    pbar.update(1)
                    try:
                        line = f.readline()
                    except:
                        continue
                    finally:
                        pos += 1

                    line = line.strip().lower()
                    if not line:
                        continue

                    tokens = []
                    for word in pku_seg.cut(line):
                        for w in self.tokenizer.tokenize(word):
                            tokens.append(w)

                    src_word = [self.vocab.get(w) for w in tokens]

                    # src_word = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]


                    if len(src_word) > self.seq_length:
                        src_word = src_word[:self.seq_length]
                    seg = [1] * len(src_word)

                    src_word, tgt = mask_seq(src_word, len(self.vocab))
                    # print('len(src_word)',len(src_word))

                    # tokens = [w for w in self.tokenizer.tokenize(line)]
                    # print([(i,a) for (i,a) in enumerate(tokens)])

                    src_pos = []
                    src_term = []
                    ## 加入pos 和terms
                    for (word, tag) in pku_seg_pos.cut(line):
                        piece_num = len(self.tokenizer.tokenize(word))
                        if word in term_set:
                            # print(word)
                            [src_term.append(1) for i in range(piece_num)]
                        else:
                            [src_term.append(0) for i in range(piece_num)]

                        [src_pos.append(pos_dict[tag]) for i in range(piece_num)]

                    if len(src_pos) > self.seq_length:
                        src_pos = src_pos[:self.seq_length]

                    # terms,labels = max_match(line.strip(),term_dict,max_num)
                    #
                    # for i,term in enumerate(terms):
                    #     if labels[i]:
                    #         for w in self.tokenizer.tokenize(term):
                    #             src_term.append(1)
                    #     else:
                    #         for w in self.tokenizer.tokenize(term):
                    #             src_term.append(0)

                    if len(src_term) > self.seq_length:
                        src_term = src_term[:self.seq_length]

                    # print('len(src_term)',len(src_term))
                    # print('src_term:',src_term)
                    # exit()


                    while len(src_word) != self.seq_length:
                        src_word.append(PAD_ID)
                        src_pos.append(pos_dict['[PAD]'])
                        src_term.append(2)
                        seg.append(PAD_ID)

                    if len(src_pos) != self.seq_length:
                        print('src_pos Problem~~~')
                        print(line)

                        tokens = [w for w in self.tokenizer.tokenize(line)]
                        print('tokens:\n', [(i, a) for (i, a) in enumerate(tokens)])

                        cut_to_tokens = []

                        for (word, tag) in pku_seg.cut(line.strip()):
                            # print(word,tag)
                            for w in self.tokenizer.tokenize(word):
                                cut_to_tokens.append(w)


                        print('cut to tokens:\n', [(i, a) for (i, a) in enumerate(cut_to_tokens)])


                        print('src_pos:\n', [(i, a) for (i, a) in enumerate(src_pos)])

                        exit()

                    if len(src_term) != self.seq_length:
                        print(line)

                        print('terms:\n',terms)
                        print('lables:\n',labels)

                        term_to_tokens = []
                        for i, term in enumerate(terms):
                            if labels[i]:
                                for w in self.tokenizer.tokenize(term):
                                    src_term.append(1)
                                    term_to_tokens.append(w)
                            else:
                                for w in self.tokenizer.tokenize(term):
                                    src_term.append(0)
                                    term_to_tokens.append(w)

                        print('term to tokens:\n',[(i,a) for (i,a) in enumerate(term_to_tokens)])

                        # tokens = [w for w in self.tokenizer.tokenize(line)]
                        print('tokens:\n',[(i,a) for (i,a) in enumerate(tokens)])
                        print('src_word:\n',[(i,a) for (i,a) in enumerate(src_word)])
                        print('src_pos:\n',[(i,a) for (i,a) in enumerate(src_pos)])
                        print('src_term:\n',[(i,a) for (i,a) in enumerate(src_term)])
                        print("tgt\n",[(i,a) for (i,a) in enumerate(tgt)])
                        print("seg\n",[(i,a) for (i,a) in enumerate(seg)])
                        exit()

                    # print(line)
                    # print('tokens:\n', [(i, a) for (i, a) in enumerate(tokens)])
                    # print('src_word:\n', [(i, a) for (i, a) in enumerate(src_word)])
                    # print('src_pos:\n', [(i, a) for (i, a) in enumerate(src_pos)])
                    # print('src_term:\n', [(i, a) for (i, a) in enumerate(src_term)])
                    # print("tgt\n", [(i, a) for (i, a) in enumerate(tgt)])
                    # print("seg\n", [(i, a) for (i, a) in enumerate(seg)])
                    # exit()


                    pickle.dump((src_word, src_pos, src_term, tgt, seg), f_write)

                    if pos >= end - 1:
                        break
                pbar.close()
        f_write.close()


class Csci_mlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src_word = []

            ## 加入pos和term
            src_pos = []
            src_term = []

            tgt = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[3])
            if masked_words_num == 0:
                continue

            for ins in instances:
                src_word.append(ins[0])

                ## 加入pos和term
                src_pos.append(ins[1])
                src_term.append(ins[2])

                seg.append(ins[4])
                tgt.append([0]*len(ins[0]))
                for mask in ins[3]:
                    tgt[-1][mask[0]] = mask[1]


            yield torch.LongTensor(src_word), \
                torch.LongTensor(src_pos), \
                torch.LongTensor(src_term), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)
