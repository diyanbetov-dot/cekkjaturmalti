import urllib.request, re
url='https://mlrs.research.um.edu.mt/CQPweb/malti04/'
req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as r:
    data=r.read().decode('utf-8', 'replace')
    print('status', r.status)
    print('content-type', r.headers.get('content-type'))
    print(data[:8000])
    print('---links---')
    for m in re.finditer(r'https?://[^\s"\']+', data):
        s=m.group(0)
        if any(k in s.lower() for k in ['malti','download','zip','gz','tar','wiki','corpus','cqpweb']):
            print(s)
