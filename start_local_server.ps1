$ErrorActionPreference = "Stop"
Set-Location "C:\Users\diyan\OneDrive\New folder\Desktop\Ċekkjatur tal-Kitba"
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe app.py *>&1 | Out-File -FilePath ".\server-live.log" -Encoding utf8
