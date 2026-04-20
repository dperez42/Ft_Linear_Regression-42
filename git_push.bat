@echo off
git add .
set /p msg="Introduce el mensaje del commit: "
rem echo "%msg%"
git commit -m "%msg%"
git push
echo Done.........