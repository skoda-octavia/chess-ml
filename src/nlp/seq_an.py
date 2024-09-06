import matplotlib.pylab as plt

with open("data/prep/full_to_tokenize.txt") as f:
    new_lines = f.readlines()

raw_seq_len = []

for line in new_lines:
    raw_seq_len.append(line.index("<PAD>"))

plt.hist(raw_seq_len, bins=40)
plt.title("Raw sequence len")
plt.ylabel("number")
plt.xlabel("seq len")
plt.show()