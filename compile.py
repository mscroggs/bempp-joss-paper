import re
import os

preamble = None

in_python = False
def to_tex(line):
    global in_python
    if in_python:
        if "```" in line:
            in_python = False
            line = line.replace("```","\\end{python}")
        else:
            return line
    if line.strip() == "```python":
        in_python = True
        return "\\begin{python}\n"
    if line[0] == "#" and line[1] != "#":
        return "\\section{"+line[1:].strip()+"}\n"
    if line[0] == "%":
        return line
    line = re.sub(r"\[@([A-Za-z0-9\-_:]+)\]", r"\\cite{\1}", line)
    line = re.sub(r"@([A-Za-z0-9\-_:]+)", r"\\cite{\1}", line)
    line = re.sub(r"``([^`]+)``", r"\\texttt{\1}", line)
    line = re.sub(r"\!\[([^\]]+)\]\(([^\)]+)\)", r"\\begin{figure}[h]\n\\centering\\includegraphics[width=.6\\textwidth]{\2}\n\\caption{\1}\\end{figure}", line)
    line = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\\underline{\1}", line)
    return line

content = ""
metadata = ""
part = ""
preamblemode = None
with open("paper.md") as f:
    for line in f:
        if line.strip() == "---":
            if preamble is None:
                preamble = True
            else:
                preamble = False
        else:
            if preamble:
                if line[:6] == "title:":
                    metadata += "\\title{"+line.split("'")[1]+"}\n"
                if line[:13] == "bibliography:":
                    bib = line[13:].split(".bib")[0].strip()
                if line[0] != " ":
                    metadata += part + "\n"
                    part = ""
                    preamblemode = line.split(":")[0]
                else:
                    if preamblemode == "tags":
                        if part == "":
                            part = "%\\tags{"
                        if part[-1] == "}":
                            part = part[:-1] + ","
                        part += line.split("-", 1)[1].strip() + "}"
                    if preamblemode == "authors":
                        if part == "":
                            part = "\\author{"
                        if "name:" in line:
                            if part[-1] == "}":
                                part = part[:-1] + ", "
                            part += line.split("name:")[1].strip() + "}"
            else:
                content += to_tex(line)

with open("paper.tex","w") as f:
    with open("preamble.tex") as f2:
        f.write(f2.read())
    f.write(metadata)
    f.write("\\begin{document}\n")
    f.write("\\maketitle\n")
    f.write(content)
    f.write("\\bibliographystyle{abbrv}\n")
    f.write("\\bibliography{"+bib+"}\n")
    f.write("\\end{document}")

os.system("pdflatex paper.tex")
os.system("pdflatex paper.tex")
os.system("bibtex paper.aux")
os.system("pdflatex paper.tex")
os.system("pdflatex paper.tex")
