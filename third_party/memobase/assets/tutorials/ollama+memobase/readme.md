`ollama` is a tool to pull and run LLMs locally on your computer

This tutorial uses `ollama` as the LLM of Memobase Server.



## Step 1

- Make sure you [install](https://ollama.com/download) the `ollama`. 

- Run `ollama -v` to see if the output is correct (my version is `0.3.8`).
- Download `qwen2.5` by running command `ollama pull qwen2.5:7b`.

> You can use any LLM you like in here, make sure it exist in ollama

## Step 2

Copy  `config_ollama.yaml.example` to `./src/server/api/config.yaml`

> [!NOTE]
>
> If you're not using `qwen2.5:7b`, you should change the field `best_llm_model` to your model in `config.yaml`

and [start Memobase server](../../../src/server/readme.md)

## Step 3

- Run `pip install memobase`
- See the results of `python ollama_memory.py`, should be something like this:

```txt
--------Use Ollama without memory--------
Q:  I'm Gus, how are you?
Hello Gus! I'm doing well, thanks for asking. How about you? How can I assist you today?
Q:  What's my name?
I'm sorry, but as an AI, I don't have any information about you personally unless you tell me your name. My main function is to assist with information and tasks, so feel free to share your name if you'd like me to address you by it!
--------Use Ollama with memory--------
Q:  I'm Gus, how are you?
Hi Gus! I'm doing well, thanks for asking. How are you today?
User Profiles: ['basic_info: name - Gus']
Q:  What's my name?
Your name is Gus.
```

