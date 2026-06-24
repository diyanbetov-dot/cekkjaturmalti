@echo off
FOR /F "tokens=5" %%a IN ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') DO (
    echo Killing process %%a on port 5000...
    taskkill /F /PID %%a
)
