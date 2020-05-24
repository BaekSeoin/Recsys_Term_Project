# -*- coding: utf-8 -*-

import content_based

def read_user_id():
    with open('input.txt', 'r') as f:
        return [int(l.strip()) for l in f.readlines()]


def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
            r = ",".join(p)
            f.write(r + "\n")
    print("5. Done")


if __name__ == "__main__":
    user_ids = read_user_id()

    ContentBased = content_based.ContentBased()
    ContentBased.genre_TF_IDF()
    ContentBased.tag_TF_IDF()
    ContentBased.cosine_similarity()
    prediction = ContentBased.recommend(user_ids)

    write_output(prediction)



