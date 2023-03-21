import os
from xml.dom.minidom import parse


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


def extract_english(other_xml, save_path):
    DOMTree = parse(other_xml)
    translations = DOMTree.documentElement
    contexts = translations.getElementsByTagName("context")
    for context in contexts:
        messages = context.getElementsByTagName("message")
        for message in messages:
            source = message.getElementsByTagName("source")[0]
            translation = message.getElementsByTagName("translation")[0]
            translation.childNodes[0] = source.childNodes[0]
    with open(save_path, "w", encoding="utf-8") as f:
        DOMTree.writexml(f, encoding="utf-8")


if __name__ == "__main__":
    source = get_source()
    target_xml = os.path.abspath("./zh_CN.xml")
    target_ts = os.path.abspath("zh_CN.ts")
    os.renames(target_xml, target_ts)
    os.system(f"pyside6-lupdate {source} -noobsolete -ts {target_ts}")
    # os.system(f"pyside6-lupdate {source} -ts {target_ts}")
    os.renames(target_ts, target_xml)

    extract_english(target_xml, os.path.abspath("./en.xml"))
