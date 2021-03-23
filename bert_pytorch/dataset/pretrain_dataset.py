from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
import os
import random
import simplejson
import argparse
import time
import multiprocessing

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

    for (i, document) in enumerate(data):
        output_fname = os.path.join(args.output_dir, "wiki-{:}".format(i*num_jobs+job_id))

        with open(output_fname, "w") as f:
            total_sentence = []
            current_sentences = [vocab["[CLS]"]]
            current_length = 1

            target_length = seq_len

            for line in document.split("\n\n"):
                line = line.strip().replace("\n", ' ')
                bert_tokens = tokenizer.tokenize(line)
                bert_tokids = [vocab[token] for token in bert_tokens]
                current_sentences += bert_tokids
                current_length += len(bert_tokids)

                if current_length >= target_length:
                    # sentence = [vocab["[CLS]"] + current_sentences[:target_length-5]
                    total_sentence.append(current_sentences)

                    if random.random() < 0.05:
                        target_length = random.randint(5, seq_len)
                    else:
                        target_length = seq_len

                    current_sentences = [vocab["[CLS]"]]
                    current_length = 1

            simplejson.dump(total_sentence, f)
            f.close()
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
        write_examples(0, wiki_data[0], args)
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


