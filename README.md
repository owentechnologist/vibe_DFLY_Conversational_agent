# vibe_DFLY_Conversational_agent
Demonstrates how a simple chat bot can leverage parts of the redis-compatible open source libraries to utilize memories and facts stored in dragonfly

# this project is a simple example of using redisvl and langgraph to implement semantic caching and capture facts and memories
# it may grow to include more agent-worflow MCP support enabled by redis/dragonfly tools and skills


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

## start the program (connects to localhost 6379 by default):

```
python3 semanticcache.py
```

## start the program with various args:

```
python3 semanticcache.py [-H host] [-p port] [-s password] [-u username] [--threshold float]

```

## example:

```
python3 semanticcache.py -H localhost -p 7900 --threshold .45
```


