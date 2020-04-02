echo 1. Activate the virtual environment && ^
%USERPROFILE%\BuildQGrain\Scripts\activate.bat && ^
echo 2. Check and install requirements && ^
pip install -r requirements.txt && ^
pip install . && ^
echo 3. Run pyinstaller to build this project && ^
pyinstaller --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\scipy\.libs ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\sklearn\.libs ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\shiboken2 ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\PySide2 ^
    --hidden-import sklearn.utils._cython_blas ^
    --icon icon.ico -w -y --clean main.py && ^
echo 4. Do some misc work && ^
cd .\dist && ^
rename main QGrain && ^
cd .. && ^
xcopy /IYE .\QGrain\i18n .\dist\QGrain\QGrain\i18n && ^
xcopy /IYE .\QGrain\settings .\dist\QGrain\QGrain\settings && ^
xcopy /IYE .\QGrain\samples .\dist\QGrain\QGrain\samples && ^
copy .\LICENSE.rtf .\dist\QGrain\ && ^
copy .\docs\tutorials\document.pdf .\dist\QGrain\document.pdf && ^
del .\dist\QGrain\QGrain\settings\chart_exporting.ini && ^
del .\dist\QGrain\QGrain\settings\QGrain.ini && ^
del .\dist\QGrain\QGrain\settings\ui.ini && ^
%USERPROFILE%\BuildQGrain\Scripts\deactivate.bat && ^
echo Finished!!!
