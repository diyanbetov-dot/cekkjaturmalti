import zipfile
p = r'C:\Users\diyan\Downloads\All.zip'
z = zipfile.ZipFile(p)
for name in z.namelist()[:3]:
    info = z.getinfo(name)
    print(name, info.compress_type, info.file_size)
    try:
        data = z.read(name)
        print(data[:80])
    except Exception as e:
        print('ERR', type(e).__name__, e)
