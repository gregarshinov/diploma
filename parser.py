import xml.etree.ElementTree as ET


class XmlIterator(object):

    def __init__(self, path_to_xml):
        self.__path = path_to_xml

    def __iter__(self):
        tree = ET.parse(self.__path)
        reviews = tree.getroot()
        for review in reviews:
            for sentence in review[0]:
                yield sentence[0].text


def main():
    for text in XmlIterator('data/se16_ru_rest_train.xml'):
        print(text)

if __name__ == '__main__':
    main()
