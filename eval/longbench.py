import argparse
import numpy
import gc
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from model_loader import load_model_and_apply_patches, add_args


ZERO_SHOT_PROMPT = "Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\nChoose the best answer by writing its corresponding letter (either A, B, C, or D)."
CHOICES = ["A", "B", "C", "D"]

def get_prompt(sample):
    #options = sample["options"]
    instruction = ZERO_SHOT_PROMPT.format(
        context=sample["context"], question=sample["question"], a=sample["choice_A"], b=sample["choice_B"], c=sample["choice_C"], d=sample["choice_D"])
    return f"{instruction}\n\nAnswer:"


def main(args):
    models = [x[0] for x in args.model]

    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("./testset/LongBench-v2", split=args.split)
    dataset = dataset.map(lambda sample: {"prompt": get_prompt(sample)})
    if args.filter_length:
        dataset = dataset.filter(lambda sample: sample["length"] == args.filter_length)

    if args.max_tokens:
        dataset = dataset.filter(lambda sample: len(
            tokenizer(sample["prompt"]).input_ids) <= args.max_tokens)
        
    if args.min_tokens:
        dataset = dataset.filter(lambda sample: len(
            tokenizer(sample["prompt"]).input_ids) >= args.min_tokens)

    choice_tokens = [x[0] for x in tokenizer(
        CHOICES, add_special_tokens=False).input_ids]
    decoded_choice = tokenizer.decode(
        choice_tokens, clean_up_tokenization_spaces=True)

    results = []
    for model in models:
        torch.cuda.empty_cache()

        loaded = load_model_and_apply_patches(model, args)

        correct_answers = 0
        i = 0
        max = len(dataset) if args.limit is None else args.limit
        
        if max ==0:
            print(f"No samples found for model {model} with the given filters.")
            break
        else:
            print(f"Evaluating {model} on {max} samples...")

        bar = tqdm(total=max)
        while i < max:
            sample = dataset[i]
            tokenized_prompt = tokenizer(sample["prompt"], return_tensors="pt")
            input_ids = tokenized_prompt.input_ids.to("cuda")
            attention_mask = tokenized_prompt.attention_mask.to("cuda")

            output = loaded.generate(input_ids, attention_mask=attention_mask,
                                    max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.eos_token_id)
            scores = output.scores[0][0]
            choice_scores = [x.cpu() for x in [scores[choice_tokens[0]],
                                            scores[choice_tokens[1]], scores[choice_tokens[2]], scores[choice_tokens[3]]]]
            selection = numpy.argmax([x.float().cpu() for x in choice_scores])

            correct_answers += 1 if CHOICES[selection] == sample["answer"] else 0

            if args.print_choices:
                print(
                    f"Choice: {CHOICES[selection]} Correct: {sample['answer']}")

            i += 1
            percent = (correct_answers / i) * 100.0

            bar.desc = f"{model}: {percent:.1f}%"
            bar.update()

            if args.aggressive_memory:
                output = None
                input_ids = None
                gc.collect()
                torch.cuda.empty_cache()

        percent = correct_answers / max
        results.append(str(percent))

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(",".join(models) + "\n")
            f.write(",".join(results) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--print-choices", action="store_true")
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--filter-length", type=str)
    parser.add_argument("--aggressive-memory", action="store_true")

    args = add_args(parser).parse_args()
    main(args)
