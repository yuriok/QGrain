import os


def get_source():
    sources = []
    for root, _, filenames in os.walk(r".\QGrain"):
        for filename in filenames:
            pure_name, extension = os.path.splitext(filename)
            if extension == ".py":
                if pure_name == "__init__":
                    continue
                sources.append(os.path.abspath(os.path.join(root, filename)))
    text = " ".join(sources)
    return text


if __name__ == "__main__":
    source = get_source()
    target_xml = os.path.abspath("./zh_CN.xml")
    target_ts = os.path.abspath("zh_CN.ts")
    os.renames(target_xml, target_ts)
    os.system(f"pyside6-lupdate {source} -noobsolete -ts {target_ts}")
    os.renames(target_ts, target_xml)
