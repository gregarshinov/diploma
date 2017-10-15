import xml.etree.ElementTree as ET
from collections import Counter
import pandas as pd


class XmlIterator(object):

    def __init__(self, path_to_xml):
        self.__path = path_to_xml

    def __iter__(self):
        tree = ET.parse(self.__path)
        reviews = tree.getroot()
        for review in reviews:
            for sentence in review[0]:
                if len(sentence) > 1 and sentence[1]:
                    polarities = []
                    for raw_opinion in sentence[1]:
                        opinion = raw_opinion.attrib
                        polarities.append(opinion['polarity'])
                    polarity = Counter(polarities).most_common(1)
                    yield (sentence[0].text, polarity[0][0])

def main():
    with open('data/test_data.tsv', 'w', encoding='utf-8') as target_file:
        for text, polarity in XmlIterator('data/RU_REST_SB1_TEST.xml'):
            target_file.write(text + '\t' + polarity + '\n')

if __name__ == '__main__':
    main()
