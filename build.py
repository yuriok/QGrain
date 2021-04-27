import os

# --windows-disable-console \
# --show-progress \
# --show-memory \
# --standalone \
# --nofollow-imports \
# --follow-import-to=QGrain \
cmd = \
"""nuitka --mingw64 \
--windows-disable-console \
--plugin-enable=pyside2 \
--enable-plugin=numpy \
--plugin-enable=torch \
--plugin-enable=pkg-resources \
--windows-icon-from-ico=icon.ico \
--output-dir=build \
QGrain_win.py"""

os.system(cmd)


