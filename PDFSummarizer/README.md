# PDF Summarizer
You'll also need to download Ollama via https://ollama.com/download for this to work when starting.


Install all these dependencies on your terminal after downloading Ollama.

```python
pip install langchain
pip install langchain-community
pip install --q unstructured langchain
pip install --q "unstructured[all-docs]"
pip install --q chromadb
pip install --q langchain-text-splitters
```

Then use the Ollama library that you downloaded.
```python
ollama pull nomic-embed-text
ollama list
```

Finally, run the python file. 
```python
python3 read.py 
```
