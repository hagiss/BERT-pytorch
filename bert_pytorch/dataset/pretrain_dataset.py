from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
import os
import random
import argparse
import time
import multiprocessing
import numpy as np

# bookcorpus = load_dataset("bookcorpus", ignore_verifications=True, split="train")
# print(bookcorpus[0])
# with open("bookcorpus.txt", "w") as f:
#     for i in bookcorpus:
#         data = i["text"]
#         data += "\n"
#         f.write(data)
#     f.close()


def write_examples(job_id, data, args):
    print("Creating example")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.get_vocab()
    seq_len = args.seq_len
    num_jobs = args.num_processes

    start_time = time.time()
    target_length = seq_len

    folder_num = 0
    folder = os.path.join(args.output_dir, "{:}".format(folder_num * num_jobs + job_id))
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_num = 0

    for (i, document) in enumerate(data):

        if file_num > 5000:
            folder_num += 1
            folder = os.path.join(args.output_dir, "{:}".format(folder_num * num_jobs + job_id))
            file_num = 0
            if not os.path.exists(folder):
                os.makedirs(folder)
        file_num += 1
        fname = os.path.join(folder, "wiki_np-{:}".format(i))
        total_sentence = []
        current_sentences = []
        current_length = 0

        for line in document.replace("\n\n", '\n').split("\n"):
            line = line.strip()
            bert_tokens = tokenizer.tokenize(line)
            bert_tokids = [vocab[token] for token in bert_tokens]
            current_sentences.append(bert_tokids)
            current_length += len(bert_tokids)

            if current_length >= target_length:
                first_segment_target_length = (target_length - 3) // 2

                first_segment = []
                second_segment = []

                for sentence in current_sentences:
                    if (len(first_segment) == 0 or
                            len(first_segment) + len(sentence) < first_segment_target_length or
                            (len(second_segment) == 0 and
                             len(first_segment) < first_segment_target_length and
                             random.random() < 0.5)):
                        first_segment += sentence
                    else:
                        second_segment += sentence



                # sentence = [vocab["[CLS]"] + current_sentences[:target_length-5]
                # total_sentence.append(current_sentences)
                first_segment = first_segment[:seq_len - 2]
                second_segment = second_segment[:max(0, seq_len - len(first_segment) - 3)]
                segments = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]] + second_segment + (
                    [vocab["[SEP]"]] if len(second_segment) > 0 else [])
                total_sentence.append(segments + [vocab["[PAD]"] for _ in range(seq_len - len(segments))])

                if random.random() < 0.05:
                    target_length = random.randint(5, seq_len)
                else:
                    target_length = seq_len

                current_sentences = []
                current_length = 0

        if current_length != 0:
            first_segment_target_length = (target_length - 3) // 2

            first_segment = []
            second_segment = []

            for sentence in current_sentences:
                if (len(first_segment) == 0 or
                        len(first_segment) + len(sentence) < first_segment_target_length or
                        (len(second_segment) == 0 and
                         len(first_segment) < first_segment_target_length and
                         random.random() < 0.5)):
                    first_segment += sentence
                else:
                    second_segment += sentence
            first_segment = first_segment[:seq_len - 2]
            second_segment = second_segment[:max(0, seq_len - len(first_segment) - 3)]
            segments = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]] + second_segment + (
                [vocab["[SEP]"]] if len(second_segment) > 0 else [])
            total_sentence.append(segments + [vocab["[PAD]"] for _ in range(seq_len - len(segments))])
        if len(total_sentence) == 0:
            raise
        # for i in total_sentence:
        #     if len(i) > 128:
        #         print(i)
        #     print(len(i))
        # print(total_sentence)
        total_sentence = np.array(total_sentence, dtype=np.int16)
        # with open(os.path.join(args.output_dir, "wiki-{:}".format(i*num_jobs+job_id)), "wb") as f:
        #     pickle.dump(total_sentence, f)
        # print(np.array(total_sentence).astype(np.int16))
        # print(total_sentence)
        np.save(fname, total_sentence)
        elapsed = time.time() - start_time
        print("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s id: {:} len: {:}".format(i+1, len(data), 100.0*(i+1)/len(data), int(elapsed), job_id, len(total_sentence)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", default=128, type=int)
    parser.add_argument("--num-processes", default=1, type=int)

    args = parser.parse_args()

    print("loading wikipedia")
    wiki = load_dataset("wikipedia", "20200501.en", ignore_verifications=True, split="train")

    wiki_data = dict.fromkeys(range(args.num_processes))

    for (i, w) in enumerate(wiki['text']):
        if wiki_data[i % args.num_processes] is None:
            wiki_data[i % args.num_processes] = []
        wiki_data[i % args.num_processes].append(w)
    print("loaded wiki datasets")

    if args.num_processes == 1:
        write_examples(0, wiki_data[0][:10], args)
    else:
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_examples, args=(i, wiki_data[i], args))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()



# for w in wiki:
#     w = wiki[10020]
#     text = w['text']
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenized = tokenizer.tokenize(text[0].strip())
    # print(text)
    # break


