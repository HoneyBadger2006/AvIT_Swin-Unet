@echo off
cd /d "C:\Users\quanp\Downloads\ISIC 2017\AViT"
"..\venv\Scripts\python.exe" -u overnight_priority_runner.py > "..\kfold_logs\overnight_priority_runner_toplevel.log" 2>&1
