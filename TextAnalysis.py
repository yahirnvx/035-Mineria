from wordcloud import WordCloud
import matplotlib.pyplot as plt

def open_file(path: str) -> str:
    content = ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.readlines()
    return " ".join(content)

all_words = ""
frase = open_file("texto.txt")
palabras = frase.rstrip().split(" ")

for arg in palabras:
    tokens = arg.split()
    all_words += " ".join(tokens) + " "

wordcloud = WordCloud(
    background_color="white",
    min_font_size=5,
    width=800,
    height=400,
    colormap='viridis'
).generate(all_words)

plt.figure(figsize=(8, 4), facecolor=None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("wordcloud_texto.png")
plt.show()
