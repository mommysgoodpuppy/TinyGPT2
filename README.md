

train.py for finetuning
```
python train.py [options]
Options:
--num_iterations [int]: Number of training iterations (default: 10).
--batch_size [int]: Batch size for training (default: 4).
--sequence_length [int]: Sequence length for training (default: 64).
--seed [int]: Random seed for reproducibility.
```
infer.py for inference
```
python infer.py [options]
Options:

--prompt [str]: Text prompt to start with (default: "What is the answer to life, the universe, and everything?").
--count [int]: Max tokens to generate (default: 100).
--temperature [float]: Temperature for sampling (default: 0.8).
--timing: Print timing per token.
--seed [int]: Random seed for reproducibility.
--batch_size [int]: Input batch size (default: 1).
--use_pretrained: Use pretrained weights instead of finetuned.
--half: Use float16 precision.
```
