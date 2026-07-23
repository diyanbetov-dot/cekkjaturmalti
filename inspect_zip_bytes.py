import os
p = r'C:\Users\diyan\Downloads\All.zip'
with open(p, 'rb') as f:
    data = f.read(64)
print('len', os.path.getsize(p))
print(data[:32])
