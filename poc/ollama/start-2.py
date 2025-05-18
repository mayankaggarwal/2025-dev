import ollama
response = ollama.list();

res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role":"user", "content":"tell me a fun fact about Mozambique"}
    ],
)
#print(res["message"]["content"]);

res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role":"user", "content":"Why is the ocean so salty?"}
    ],
    stream=True,
)

#for chunk in res:
#    print(chunk["message"]["content"],end="",flush=True)

#print(ollama.show("llama3.2"));

#Create a new model with modelfile

modelfile1 = """
FROM llama3.2
SYSTEM You are James, a very smart assistant who knows everthing about oceans, You are very succint and informative.
PARAMETER temperature 0.1
"""

ollama.create(model="knowitall", modelfile=modelfile1)
res = ollama.generate(model="knowitall",prompt="why is ocean so salty?")
print(res["response"])



