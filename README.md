# LangChain learning 

Shared model lives in `src/langchain_demo/llm.py`.  
Each concept is a separate script under `lessons/`.

## Layout

```
langchain/
  .env                          # ANTHROPIC_API_KEY=...
  main.py                       # optional lesson picker
  lessons/
    01_basic_chat.py
    02_prompt_templates.py
    ...                         # add more as you learn
  src/langchain_demo/
    llm.py                      # get_llm() — reuse everywhere
```

