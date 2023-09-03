@echo off
echo Activating conda adam_mmlab environment...
call conda activate adam_mmlab

REM Wait 5s
ping 127.0.0.1 -n 5 > nul

echo Running pyinstaller command...
pyinstaller --noconfirm --onedir --noconsole --clean --add-data "alg/assets;alg/assets/" --add-data "alg/model;alg/model/" --add-data "D:/Anaconda/envs/adam_mmlab/Lib/site-packages/yapf_third_party;yapf_third_party/" --hidden-import "mmcv._ext"  "DeepCell.py"
