<p align="center">
  <h1 style="display: inline;">
    <img src="./assets/MMVU_logo.jpg" alt="Logo" style="width: 50px; vertical-align: middle; margin-right: 10px;">
    MMVU: Measuring Expert-Level Multi-Discipline Video Understanding
  </h1>
</p>

<p align="center">
  <a href="https://mmvu-benchmark.github.io/">🌐 Homepage</a> •
  <a href="https://mmvu-benchmark.github.io/#leaderboard">🥇 Leaderboard</a> •
  <a href="https://arxiv.org/abs/2501.12380">📖 Paper</a> •
  <a href="https://huggingface.co/datasets/yale-nlp/MMVU">🤗 Data</a>
</p>


## 📰 News

- **2025-01-21**: We are excited to release the MMVU paper, dataset, and evaluation code!

## 👋 Overview
![Local Image](./assets/overview.png)

### Why MMVU Benchmark?
Despite the rapid progress of foundation models in both text-based and image-based expert reasoning, there is a clear gap in evaluating these models’ capabilities in **specialized-domain video** understanding. Videos inherently capture **temporal dynamics**, **procedural knowledge**, and **complex interactions**—all of which are crucial for expert-level tasks across disciplines like healthcare, engineering, and scientific research. Unlike static images or text, specialized-domain videos often require integrating **domain-specific expertise** (e.g., understanding chemical reactions, medical procedures, or engineering workflows) alongside traditional **visual perception**.

MMVU is designed to **bridge this gap** and offer a **multidisciplinary** perspective by providing:
   - **3,000 expert-annotated QA examples** spanning **1,529 specialized-domain videos** across **27 subjects** in **four key disciplines** (Science, Healthcare, Humanities & Social Sciences, and Engineering).  
   - Ensures both **breadth of domain knowledge** and **depth of reasoning**, reflecting real-world complexities in specialized fields.
   - Each example comes with **expert-annotated reasoning rationales** and **relevant domain knowledge**, enabling researchers to assess not just **answer correctness** but also **reasoning quality**.  

## 🚀 Quickstart
### 1. Setup
Install the required packages and Setup up `.env` file
```bash
pip install -r requirements.txt
```


**Dataset Example Feature**:
```bash
{
    "id": // Unique ID for the question
    "video": // HF download link to the video
    "youtube_url": // original Youtube URL to the video
    "question_type": // "open-ended" or "multiple-choice"
    "metadata": {
        "subject": // Subject of the example
        "textbook": // From which textbook the example is curated from
        "rationale": // rationale for the answer (Coming Soon!)
        "knowledge": // List of wikipedia URLs for the domain knowledge (Coming Soon!)
    },
    "question":  // The question
    "choices": // choices for multiple-choice questions
    "answer": // answer to the question
},
```


### 2. Response Generation
As detailed in Appendix B.1, we evaluate models using three different types of model inference: API-based, vllm, and HuggingFace, depending on the specific model's availability. To generate responses for the MMVU validation set, run the following command:
```bash
bash model_inference_scripts/run_api_models.sh # Run all API models
bash model_inference_scripts/run_hf_models.sh # Run model inference using HuggingFace
bash model_inference_scripts/run_vllm_image_models.sh # Run model that supports multi-image input using vllm
bash model_inference_scripts/run_vllm_video_models.sh # Run model that supports video input using vllm
```

The generated responses will be saved in the `outputs/validation_{prompt}` directory. Where `{prompt}` is `cot` for CoT reasoning and `direct-output` for direct answering without intermediate reasoning steps.

### 3. Evaluation
To evaluate the generated responses, run the following command:
```bash
python acc_evaluation.py --output_dir <output_dir>
```
The evaluation results will be saved in the `outputs/evaluation_results/` directory.


## 📋 Results from Existing Models
We release full results on the validation set (i.e., generated responses, accuracy measurement done by GPT-4o) for all models we tested in our [HuggingFace Repo (Coming Soon!)](https://huggingface.co/datasets/yale-nlp/MMVU_model_outputs). If you are interested in doing some fine-grained analysis on these results, feel free to use them!

## 🥇 Leaderboard Submission
The MMVU test set remains hidden from the public to minimize data contamination and ensure an unbiased evaluation of model capabilities. We are developing an online submission system for the leaderboard. 
In the meantime, if you would like to evaluate your model or method on the MMVU test set before the submission system becomes available, please reach out to Yilun Zhao at yilun.zhao@yale.edu and share the codebase you used to generate results on the validation set. We will run your model on the test set and provide you with the evaluation results. You could then decide whether to update your results to the leaderboard.

## ✍️ Citation
If you use our work and are inspired by our work, please consider cite us (available soon):
```
@misc{zhao2025mmvu,
      title={MMVU: Measuring Expert-Level Multi-Discipline Video Understanding}, 
      author={Yilun Zhao and Lujing Xie and Haowei Zhang and Guo Gan and Yitao Long and Zhiyuan Hu and Tongyan Hu and Weiyuan Chen and Chuhan Li and Junyang Song and Zhijian Xu and Chengye Wang and Weifeng Pan and Ziyao Shangguan and Xiangru Tang and Zhenwen Liang and Yixin Liu and Chen Zhao and Arman Cohan},
      year={2025},
      eprint={2501.12380},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.12380}, 
}
```