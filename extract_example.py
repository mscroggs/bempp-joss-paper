output = []

with open("paper.md") as f:
    for part in f.read().split("```python")[1:]:
        output.append(part.split("```")[0])

with open("example.py", "w") as f:
    f.write("\n\n".join(output))
