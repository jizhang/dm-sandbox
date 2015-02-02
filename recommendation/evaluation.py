import math

def popularity(train, test, N):
    item_popularity = {}
    for user, items in train.items():
        for item in items.keys():
            item_popularity[item] = item_popularity.get(item, 0) + 1

    ret = 0
    n = 0
    for user in train.keys():
        rank = get_recommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1

    ret /= float(n)
    return ret
