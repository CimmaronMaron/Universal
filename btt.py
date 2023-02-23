def bt(n):
    if n == '':
        n=n+'X'
    import csv
    data = csv.StringIO()
    with open(data) as dt:
        data.write(n.encode('ut8-8'))
    return data
