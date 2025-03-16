
# LightTransfer

<div align="center">

<img src="https://github.com/sail-sg/LightTrans/blob/main/Figure/LightTransfer_logo.jpg" width="200"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation</font></b>
    <sup>
      <a href="https://sites.google.com/view/lighttransfer">
      </a>
    </sup>
    <div>&nbsp;</div>
  </div>
  
[🤗HuggingFace](https://huggingface.co/cxdu/QwQ-32B-LightTransfer) | • [🆕Update News](#news) | • [🤔Reporting Issues](https://github.com/sail-sg/LightTrans/issues/new) |  • [📜Paper Link](https://arxiv.org/pdf/2410.13846)

</div>

## Introduction  

**LightTransfer** is a lightweight transformation framework for enhancing the efficiency of large transformer models, such as LLaMA and QwQ, in long-context understanding and long CoT generation. By identifying **lazy layers**—those primarily attending to initial or recent tokens—LightTransfer replaces their full attention with streaming attention, significantly reducing memory overhead.  

- **Improved efficiency with minimal performance loss**:  
  LightTransfer achieves up to **2.17× higher throughput** while maintaining strong performance (<1.5% drop on LongBench).  
- **Flexible adaptation for long-context tasks**:  
  Works **without retraining** for long-context understanding and requires only minimal fine-tuning for advanced long CoT generation, such as mathematical reasoning in **QwQ-STILL**, achieving **53.3% on AIME24**.  

For more details, visit our [project page](https://sites.google.com/view/lighttransfer).

## News

\[2025.03.16\] We release the checkpoint of QwQ-32B-LightTransfer. See [model card](https://huggingface.co/cxdu/QwQ-32B-LightTransfer) for details.

## LightTranfer-Train

We release the checkpoint of **QwQ-LightTransfer**, which is a 32B-parameter model built on **Qwen/Qwen2.5-32B-Instruct** and fine-tuned via SFT on **RUC-AIBOX/long_form_thought_data_5k**. 
- By replacing 50% of the model’s full attention layers with streaming attention,specifically layers [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 37, 38, 43, 51], it substantially reduces memory costs.
- QwQ-LightTransfer scores 53.3% on the advanced math benchmark AIME24, demonstrating its strong o1-like long reasoning capabilities.

### Performance Evaluation
We have evaluated QwQ-LightTransfer on several long reasoning generation benchmarks. Some of the evaluation results are shown in the table below.
| Method         | Math-OAI | AIME24 | AIME25 | GSM8K |
|---------------|---------|--------|--------|-------|
| o1-preview    | 85.5    | 44.6   | -      | -     |
| OwO-STILL     | 90.2    | 46.7   | 33.3   | 95.6  |
| LongGen       | 78.2    | 16.7   | -      | 95.4  |
| LightTransfer | 90.7    | 53.3   | 40.0   | 95.5  |

### Usages

**_Import from Transformers_**

To load the QwQ-LightTransfer model using Transformers, use the following code:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'QwQ-32B-LightTransfer'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,trust_remote_code=True,device_map='auto')

text = "Hi, I'm QwQ-32B-LightTransfer."
inputs = tokenizer(text, return_tensors='pt').to(model.device)
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'],max_gen_len=32000)
print(tokenizer.decode(outputs[0]))
```


_**Evaluation scripts**_

## License

Code and model weights are licensed under Apache-2.0.

## Citation
```
@misc{zhang2025lighttransferlongcontextllmsecretly,
      title={LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation}, 
      author={Xuan Zhang and Fengzhuo Zhang and Cunxiao Du and Chao Du and Tianyu Pang and Wei Gao and Min Lin},
      year={2025},
      eprint={2410.13846},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.13846}, 
}
```
