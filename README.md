# vibe_DFLY_Conversational_agent
Demonstrates how a simple chat bot can leverage parts of the redis-compatible open source libraries to utilize memories and facts stored in dragonfly

# this project is a simple example of using redisvl and langgraph to implement semantic caching and capture facts and memories
# it may grow to include more agent-worflow MCP support enabled by redis/dragonfly tools and skills

# the application uses localai by default

For installation and management information see: 
https://github.com/mudler/LocalAI 

```
LocalAI: Built primarily in Go and utilizing Docker containers, it is an enterprise-grade backend. It functions as a drop-in replacement for OpenAI API, allowing developers to plug local text, audio, and image generation models into existing apps.

LocalAI: Cross-platform and hardware-agnostic. It can be deployed via Docker on Linux, Windows, or macOS, and supports both CPUs and Nvidia GPUs.

Best for developers, sysadmins, or teams who want to self-host an entire API infrastructure, run AI in CI/CD pipelines, or integrate local voice and image features into their own software.
```

## in agent.py you will find the setup for localai and the embedding model for semantic cache use:

Adjust these to suit your deployment / preferred LLM etc

```
LLM_BASE_URL = "http://localhost:6060/v1"
LLM_MODEL = "qwen3.5-9b-glm5.1-distill-v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMS = 768  # all-mpnet-base-v2 output dimension
CACHE_INDEX_NAME = "llm_semantic_cache"
EXTRACTION_EVERY_N = 3
```


# Python Environment Setup 
- **A: Create a virtual environment:**

```
python3 -m venv lcenv
```

- **B. Activate it:  [This step is repeated anytime you want this venv back]**

```
source lcenv/bin/activate
```

On windows you would do:

```
lcenv\Scripts\activate
```
If no permission in Windows
 The Fix (Temporary, Safe, Local):
In PowerShell as Administrator, run:
```

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then confirm with Y when prompted.


- **C. Install the libraries: [only necesary to do this one time per environment]**

```
pip3 install -r requirements.txt
```


## start the program with various args:

```
python3 app.py [-H host] [-p port] [-s password] [-u username] [--threshold float] --web-port port

```

## example where the webapp is available on port 9026 and it connects to DF on port 7900:

```
 python3  app.py --web-port 9026 -H localhost -p 7900
 
 ```


