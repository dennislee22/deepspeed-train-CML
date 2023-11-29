import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    T5ForConditionalGeneration,
    set_seed,
)
from datasets import load_from_disk
import torch
import evaluate
import nltk
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import TrainerCallback, EarlyStoppingCallback, is_tensorboard_available
import os

nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")
# evaluation generation args
gen_kwargs = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 50,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
}

def remove_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
        print(f"Folder '{dir_path}' has been deleted.")
    except Exception as e:
        # Ignore errors, you can print a message if needed
        print(f"Folder '{dir_path}' has been deleted.")
        
def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="t5-small", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="my_dataset", help="Path to the already processed dataset.")
    parser.add_argument("--outputdir", type=str, default="trainoutput-wikisql", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=128, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args
def custom_rewrite_logs(d, mode):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if mode == 'eval' and k.startswith(eval_prefix):
            if k[eval_prefix_len:] == 'loss':
                new_d["combined/" + k[eval_prefix_len:]] = v
        elif mode == 'test' and k.startswith(test_prefix):
            if k[test_prefix_len:] == 'loss':
                new_d["combined/" + k[test_prefix_len:]] = v
        elif mode == 'train':
            if k == 'loss':
                new_d["combined/" + k] = v
    return new_d


class CombinedTensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).
    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writers=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writers = tb_writers

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writers = dict(train=self._SummaryWriter(log_dir=os.path.join(log_dir, 'train')),
                                   eval=self._SummaryWriter(log_dir=os.path.join(log_dir, 'eval')))

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writers is None:
            self._init_summary_writer(args, log_dir)

        for k, tbw in self.tb_writers.items():
            tbw.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    tbw.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(tbw, "add_hparams"):
                tbw.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writers is None:
            self._init_summary_writer(args)

        for tbk, tbw in self.tb_writers.items():
            logs_new = custom_rewrite_logs(logs, mode=tbk)
            for k, v in logs_new.items():
                if isinstance(v, (int, float)):
                    tbw.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            tbw.flush()

    def on_train_end(self, args, state, control, **kwargs):
        for tbw in self.tb_writers.values():
            tbw.close()
        self.tb_writers = None
        
def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # load model from the hub
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    remove_dir(args.outputdir) 
    output_dir = args.outputdir.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        report_to="tensorboard",
    )
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience= 5, 
                                    early_stopping_threshold= 0.001)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks= [early_stopping,CombinedTensorBoardCallback]
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)
    trainer.create_model_card()


def main():
    args, _ = parse_arge()
    training_function(args)

if __name__ == "__main__":
    main()