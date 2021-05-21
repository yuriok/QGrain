
import os
import sys

PRO_FILENAME = r"./QGrain.pro"

PRO_TEMPLATE = \
"""
SOURCES={0}
TRANSLATIONS=en.ts zh_CN.ts
CODECFORTR=UTF-8
"""

def update_qt_pro():
    sources = []
    for root, _, filenames in os.walk("./QGrain"):
        for filename in filenames:
            if os.path.splitext(filename)[-1] == ".py":
                sources.append(os.path.abspath(os.path.join(root, filename)))
    text = "\\\n    ".join(sources)
    with open(PRO_FILENAME, "w") as f:
        f.write(PRO_TEMPLATE.format(text))

if __name__ == "__main__":
    update_qt_pro()
    os.system("pyside2-lupdate -noobsolete ./QGrain.pro")
