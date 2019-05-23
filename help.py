#따옴표를 없애고 앞, 뒤의 공백을 제거
a = []

for x in a:
    x.strip()
    d=""
    for v in x:
        if(not v=='\''):d = d+v
    print(d+",",sep='')