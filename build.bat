echo 1. Activate the virtual environment && ^
%USERPROFILE%\BuildQGrain\Scripts\activate.bat && ^
echo 2. Change to the work directory && ^
cd %USERPROFILE%\Desktop\QGrain && ^
echo 3. Check and install requirements && ^
pip install -r %USERPROFILE%\Desktop\QGrain\requirements.txt && ^
pip install . && ^
echo 4. Run pyinstaller to build this project && ^
pyinstaller --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\scipy\.libs ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\sklearn\.libs ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\shiboken2 ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\PySide2 ^
    --hidden-import sklearn.utils._cython_blas ^
    --icon icon.ico -w -y --clean main.py && ^
echo 5. Do some misc work && ^
cd %USERPROFILE%\Desktop\QGrain\dist && ^
rename main QGrain && ^
cd QGrain && ^
xcopy /IYE %USERPROFILE%\Desktop\QGrain\QGrain\i18n %USERPROFILE%\Desktop\QGrain\dist\QGrain\QGrain\i18n && ^
xcopy /IYE %USERPROFILE%\Desktop\QGrain\QGrain\settings %USERPROFILE%\Desktop\QGrain\dist\QGrain\QGrain\settings && ^
xcopy /IYE %USERPROFILE%\Desktop\QGrain\QGrain\samples %USERPROFILE%\Desktop\QGrain\dist\QGrain\QGrain\samples && ^
copy %USERPROFILE%\Desktop\QGrain\LICENSE.rtf %USERPROFILE%\Desktop\QGrain\dist\QGrain\ && ^
copy %USERPROFILE%\Desktop\QGrain\docs\tutorials\document.pdf %USERPROFILE%\Desktop\QGrain\dist\QGrain\document.pdf && ^
del %USERPROFILE%\Desktop\QGrain\dist\QGrain\QGrain\settings\chart_exporting.ini && ^
del %USERPROFILE%\Desktop\QGrain\dist\QGrain\QGrain\settings\QGrain.ini && ^
del %USERPROFILE%\Desktop\QGrain\dist\QGrain\QGrain\settings\ui.ini && ^
cd %USERPROFILE%\Desktop\QGrain && ^
%USERPROFILE%\BuildQGrain\Scripts\deactivate.bat && ^
echo Finished!!!
