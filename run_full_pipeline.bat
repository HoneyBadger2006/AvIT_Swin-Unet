@echo off
cd /d "C:\Users\quanp\Downloads\ISIC 2017\AViT"
"C:\Users\quanp\Downloads\ISIC 2017\venv\Scripts\python.exe" -u pipeline_runner.py --phase all >> "C:\Users\quanp\Downloads\ISIC 2017\kfold_logs\pipeline_all_stdout.log" 2>&1
