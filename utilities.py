def prod(list):
    res = list[0]
    for i in range(1,len(list)):
        res = res * list[i]
    return res