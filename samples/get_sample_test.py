from random import sample, choice, random, randint

words = []
with open("serch_words.csv", 'r', encoding='utf-8') as fr:
    for word in fr:
        if word:
            words.append(word[:-1])

with open("..\\data\\ali_ccp_sample_train_1k.csv", 'r', encoding='utf-8') as fr:
    with open("..\\data\\sample_test.csv", 'w', encoding='utf-8') as fw:
        for line in fr:
            if not line:
                continue
            line = line.split(",")
            new = [line[0], line[1], line[2], line[3], str(round(random(), 5)),
                   chr(1).join([chr(2).join(tup.split(":")) for tup in line[4].split("|")[:randint(0, 7)]]),
                   chr(1).join(line[19].split("|")[:randint(0, 7)]),
                   choice(words),
                   chr(1).join([word + chr(2) + str(round(random(), 5)) for word in sample(words, randint(0, 7))]),
                   chr(1).join(sample(words, randint(0, 7)))
                   ]
            new = ",".join(new)+"\n"
            fw.write(new)
