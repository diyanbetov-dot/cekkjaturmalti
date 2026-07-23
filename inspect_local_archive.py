import os, zipfile
path = r'C:\Users\diyan\Downloads\All.zip'
print('exists', os.path.exists(path))
if os.path.exists(path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        print('count', len(names))
        for name in names[:80]:
            print(name)
