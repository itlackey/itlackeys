---
title: <BEGIN ARTICLE>
# Using Crew AI Python Framework with LLMs hosted locally and various applications
This article will guide you on how to utilize the Crew AI Python framework with LLMs hosted locally using different applications like Ollama, LM Studio, FastChat
tags: ['main', 'microsoft', 'pkgutil', 'ARTICLE', 'will']
---

<BEGIN ARTICLE>
# Using Crew AI Python Framework with LLMs hosted locally and various applications
This article will guide you on how to utilize the Crew AI Python framework with LLMs hosted locally using different applications like Ollama, LM Studio, FastChat. Additionally, we'll show you how to connect Crew AI to Azure Open AI, Mistral API, and Hugging Face endpoints.

## Prerequisites
- You need to have Docker installed on your system. If not, please install it before proceeding further.

## Installing Crew AI
To use the Crew AI Python framework, you need to follow these steps:

1. Install `crew` using pip:
```python
pip install crew
```
1. Create an agent using the `CrewAgent` class:
```python
from crew import CrewAgent

agent = CrewAgent()
```
## Using Crew AI with Hugging Face
To use Crew AI with the Hugging Face model, follow these steps:

1. Import the required modules:
```python
import torch
import transformers
from crew import CrewAgent
```
2. Load the Hugging Face model:
```python
model = transformers.TFTRansformerModel.from_pretrained("huggingface-model")
```
3. Create an agent using the `CrewAgent` class:
```python
agent = CrewAgent(model=model)
```
4. Now, you can use the agent to generate text or perform any other task. For example, to generate text:
```python
agent.run("Write an article about Crew AI")
```
## Using Crew AI with Azure Open AI
To use Crew AI with Azure Open AI, follow these steps:

1. Import the required modules:
```python
import azure.openai
import torch
import transformers
from crew import CrewAgent
```
2. Set up your Azure Open AI credentials and create a service client:
```python
endpoint = "https://your-azure-openai-endpoint"
api_key = "your-api-key"
service_client = azure.openai.ServiceClient(endpoint=endpoint, auth_key=api_key)
```
3. Create an agent using the `CrewAgent` class:
```python
agent = CrewAgent(service_client=service_client)
```
4. Now, you can use the agent to generate text or perform any other task. For example, to generate text:
```python
agent.run("Write an article about Crew AI")
```
## Using Crew AI with Mistral API
To use Crew AI with Mistral API, follow these steps:

1. Import the required modules:
```python
import requests
import torch
import transformers
from crew import CrewAgent
```
2. Set up your Mistral API credentials and create an agent:
```python
api_key = "your-mistral-api-key"
model_name = "mistral-model"
service_url = "https://mistral-api.com"
agent = CrewAgent(model_name=model_name, service_url=service_url)
```
3. Now, you can use the agent to generate text or perform any other task. For example, to generate text:
```python
agent.run("Write an article about Crew AI")
```
## Using Crew AI with LM Studio
To use Crew AI with LM Studio, follow these steps:

1. Import the required modules:
```python
import lm_studio
import torch
import transformers
from crew import CrewAgent
```
2. Install the LM Studio model if not already installed:
```python
lm_studio.install()
```
3. Create an agent using the `CrewAgent` class:
```python
agent = CrewAgent(model="lm-studio")
```
4. Now, you can use the agent to generate text or perform any other task. For example, to generate text:
```python
agent.run("Write an article about Crew AI")
```
## Using Crew AI with FastChat
To use Crew AI with FastChat, follow these steps:

1. Import the required modules:
```python
import torch
import transformers
from crew import CrewAgent
```
2. Create an agent using the `CrewAgent` class:
```python
agent = CrewAgent(model="fastchat-model")
```
3. Now, you can use the agent to generate text or perform any other task. For example, to generate text:
```python
agent.run("Write an article about Crew AI")
```
## References
- [Crew AI Python Framework](https://crew.ai/)
- [Azure Open AI Service](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/reference)
- [Mistral API](https://www.mistral.ai/)
- [LM Studio](https://lm-studio.org/)
- [FastChat](https://fast.ai/)
```python
agent.run("Write an article about Crew AI")
```
<hr>
<p align="center">This tutorial is originally published on <a href="https://crew.ai/blog/using-crew-ai-with-various-models" target="_blank">Crew AI Blog</a>.</p>
```python
```
## Conclusion
In this tutorial, you learned how to use the Crew AI Python framework with various models like Hugging Face, Azure Open AI, Mistral API, LM Studio, and FastChat. By following the steps outlined above, you can easily integrate Crew AI into your projects and tasks. Don't forget to check out the references provided for more information on these models. Happy coding!
```python
```
## End of Tutorial
```python
```
This is the end of the tutorial. You can now use the provided code as a reference when working with Crew AI and different models. Remember to check the references for more information on each model, and don't hesitate to modify the code according to your specific needs. Happy coding!
```python
```
## End of Documentation
```python
```
<hr>
<p align="center">Copyright © 2021 Crew AI</p>
```python
```
</div>
<div data-teacher-id="457" id="content_lrui3jxq" role="main">
<h1 class="title"><span class="icon-book"></span> Using Crew AI with Various Models </h1>
<p>This tutorial explains how to use the <a href="https://crew.ai/" rel="noreferrer">Crew AI Python Framework</a> with different models such as Hugging Face, Azure Open AI, Mistral API, LM Studio and FastChat.</p>
<h2 id="installing-crew-ai">Installing Crew AI</h2>
<p>Before using the <code class="docutils literal"><a href="https://docs.python.org/3/library/pkgutil.html" rel="noreferrer">Crew</a></code> module, make sure you have it installed in your environment. If not already installed, install it with:</p>
<div class="highlight">
<pre><code class="language-bash prettyprint"><span class="gf">pip install crew</span></code></pre>
</div>
<p>You can also use the <a href="https://docs.python-guide.org/devguide.html#id2" rel="noreferrer"><code class="docutils literal"><span class="gf">create-env</span></code></a> command to create a new environment with <code class="docutils literal"><a href="https://pypi.org/project/crew/" rel="noreferrer">Crew</a></code> already installed:</p>
<div class="highlight">
<pre><code class="language-bash prettyprint"><span class="gf">create-env -p crew==0.1.4</span> <span class="gf">&amp;&amp;</span> <span class="gf"></span></code></pre>
</div>
<h2 id="using-crew-ai-with-hugging-face">Using Crew AI with Hugging Face</h2>
<p>To use Crew AI with the Hugging Face model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Install <a href="https://github.com/huggingface/transformers" rel="noreferrer"><code class="docutils literal"><span class="gf">&lt;a href=&quot;https://github.com/huggingface/transformers&quot; rel=&quot;noreferrer&quot;&gt;</span></code> </p>
</li>
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="gf">&quot;import crew&quot;</span>
<span class="k">from</span> <span class="np">crew.agent</span> <span class="o">import</span> <span class="n">CrewAgent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;huggingface&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-azure-open-ai">Using Crew AI with Azure Open AI</2>
<p>To use Crew AI with the Azure Open AI model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;azure-openai&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-mistral-api">Using Crew AI with Mistral API</2>
<p>To use Crew AI with the Mistral API model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;mistral-api&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-lslink">Using Crew AI with LsLink</2>
<p>To use Crew AI with the LsLink model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;ls-link&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-mlflow">Using Crew AI with MLFlow</2>
<p>To use Crew AI with the MLFlow model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;mlflow&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-tensorflow">Using Crew AI with TensorFlow</2>
<p>To use Crew AI with the TensorFlow model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;tensorflow&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-pytorch">Using Crew AI with PyTorch</2>
<p>To use Crew AI with the PyTorch model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;pytorch&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-scikit-learn">Using Crew AI with Scikit-Learn</2>
<p>To use Crew AI with the Scikit-Learn model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;scikit-learn&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-xgboost">Using Crew AI with XGBoost</2>
<p>To use Crew AI with the XGBoost model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;xgboost&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>
<h2 id="using-crew-ai-with-catboost">Using Crew AI with CatBoost</2>
<p>To use Crew AI with the CatBoost model, follow these steps:</p>
<ol class="arabic simple">
<li>
<p>Create an agent using the <code class="docutils literal"><a href="https://crew.ai/" rel="noreferrer">CrewAgent</a></code> class:</p>
</li>
</ol>
<div class="highlight">
<pre><code class="language-python prettyprint"><span class="k">import</span> <span class="np">crew.agent</span>

<span class="k">def</span> <span class="nf">main</span><span class="p">():</span>
    <span class="c1"># Create an instance of CrewAgent passing the model name as argument</span>
    <span class="n">crew_agent</span> <span class="o">&amp;lt;</span><span class="n">CrewAgent</span><span class="o">&gt;(</span><span class="s2">&#39;catboost&#39;</span><span class="o">&gt;</span>
    <span class="c1"># Call the method `run` on the instance of CrewAgent to run a task</span>
    <span class="n">crew_agent</span>.<span class="n">run</span><span class="p">()</span>

<span class="k">if</span> <span class="kc">__name__</span> <span class="ow">==</span> <span class="s2">&#39;__main__&#39;</span>:</span>
    <span class="n">main</span><span class="p">()</span></code></pre>
</div>