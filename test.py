def func(a, b, c, **kwargs):
    print(a, b, c)

func(g=22, a=1, b=2, c=3, d=4)
args = {"g":22, "a":1, "k":0, "b":2, "c":3, "d":4}
func(**args)