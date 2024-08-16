# GPT2 with Flax NNX
This is an implementation of GPT2 based on [karpathy/nanoGPT](https://github.com/karpathy/nanogpt) in Google's FLAX NNX.

The Jupyter notebooks describe the ordeal!

#### Dependencies

pip install -r requirements.txt

Tested with `Python 3.9`

#### Usage

##### Generation
Generation with single prompt and multiple prompts is possible. 

The generations while quite coherent in terms of language doesn't look that great, I believe this is what GPT2 is capable of with just pre-training. Instruct tuning would make it better, I expect. 

```
$ python generate.py \
        -p "I am a teacher," \
        -p "Capital of India is" \
        -p "I am a Doctor and my job is to" \
        -m "gpt2-xl" -n 50
        
loading weights from pretrained gpt: gpt2-xl
Length of pytorch state dict: 581
Length of prepared JAX modules dict: 292
Total JAX matrices: 581
Transposing  lm_head
> I am a teacher, a nurse and a coach at my own gym. The only thing to do with my life is to help others."
Papadopoulos' father, Michael Papadopoulos, is a Greek
> Capital of India is considered the first government in the world to be governed by an all-party committee, which was set up in May 2005 during a three-day debate on 'The Future of Education under the People's
> I am a Doctor and my job is to treat people and the disease it seeks to control. Some have said I have no idea what I am doing. But we all agree that we should have strong, good evidence, backed up by good science
```

```
$ python generate.py -p "I am a teacher," -m "gpt2" -r 2
loading weights from pretrained gpt: gpt2
Length of pytorch state dict: 149
Length of prepared JAX modules dict: 76
Total JAX matrices: 149
Transposing  lm_head

> I am a teacher, an educator and a teacher, but I will always be grateful if you will support and support my work here and for everything that
> I am a teacher, but please support me. Let me teach. Yes I have this job. Please pay it forward. That's it. I
```

##### Benchmark with HellaSwag
The numbers match known HellaSwag results.
```
$ python hellaswag.py -m "gpt2-xl"
loading weights from pretrained gpt: gpt2-xl
Length of pytorch state dict: 581
Length of prepared JAX modules dict: 292
Total JAX matrices: 581
Transposing  lm_head
1 acc_norm: 0/1=0.0000
---
Context:
 A man is sitting on a roof. he
Endings:
0 (loss: 3.7495) is using wrap to wrap a pair of skis.
1 (loss: 5.4112) is ripping level tiles off.
2 (loss: 2.3814) is holding a rubik's cube.
3 (loss: 4.1588) starts pulling up roofing on a roof.
predicted: 2, actual: 3
2 acc_norm: 1/2=0.5000

...

40 acc_norm: 13/40=0.3250
41 acc_norm: 14/41=0.3415
42 acc_norm: 15/42=0.3571
43 acc_norm: 16/43=0.3721
44 acc_norm: 17/44=0.3864
45 acc_norm: 18/45=0.4000
46 acc_norm: 19/46=0.4130
47 acc_norm: 20/47=0.4255
48 acc_norm: 21/48=0.4375
49 acc_norm: 21/49=0.4286
50 acc_norm: 22/50=0.4400
```

##### References
* https://github.com/karpathy/nanoGPT
* https://github.com/cgarciae/nanoGPT-jax
* https://github.com/jenkspt/gpt-jax