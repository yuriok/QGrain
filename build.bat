echo 1. Activate the virtual environment && ^
%USERPROFILE%\BuildQGrain\Scripts\activate.bat && ^
echo 2. Change to the work directory && ^
cd %USERPROFILE%\Desktop\QGrain && ^
echo 3. Check and install requirements && ^
pip install -r %USERPROFILE%\Desktop\QGrain\requirements.txt && ^
echo 4. Run pyinstaller to build this project && ^
pyinstaller --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\scipy\.libs ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\sklearn\.libs ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\shiboken2 ^
    --paths %USERPROFILE%\BuildQGrain\Lib\site-packages\PySide2 ^
    --hidden-import sklearn.utils._cython_blas ^
    --icon icon.ico -w -y --clean qgrain.py && ^
echo 5. Do some misc work && ^
cd %USERPROFILE%\Desktop\QGrain\dist\ && ^
rename qgrain QGrain && ^
cd QGrain && ^
rename qgrain.exe QGrain.exe && ^
rename qgrain.exe.manifest QGrain.exe.manifest && ^
xcopy /IYE %USERPROFILE%\Desktop\QGrain\i18n %USERPROFILE%\Desktop\QGrain\dist\QGrain\i18n && ^
xcopy /IYE %USERPROFILE%\Desktop\QGrain\settings %USERPROFILE%\Desktop\QGrain\dist\QGrain\settings && ^
xcopy /IYE %USERPROFILE%\Desktop\QGrain\samples %USERPROFILE%\Desktop\QGrain\dist\QGrain\samples && ^
copy %USERPROFILE%\Desktop\QGrain\LICENSE.rtf %USERPROFILE%\Desktop\QGrain\dist\QGrain\ && ^
copy %USERPROFILE%\Desktop\QGrain\docs\tutorials\document.pdf %USERPROFILE%\Desktop\QGrain\dist\QGrain\document.pdf && ^
del %USERPROFILE%\Desktop\QGrain\dist\QGrain\settings\chart_exporting.ini && ^
del %USERPROFILE%\Desktop\QGrain\dist\QGrain\settings\QGrain.ini && ^
del %USERPROFILE%\Desktop\QGrain\dist\QGrain\settings\ui.ini && ^
echo 6. back to desktop directory && ^
cd %USERPROFILE%\Desktop\QGrain && ^
echo Finished!!!
