import pandas as pd

from src.NLU import NaturalLanguageUnderstanding
from src.CRAWLER import SearchCrawler
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

class NaturalLanguageGenerator:
    def __init__(self):
        self.template_dir = "./dataset/search_template_dataset.csv"
        self.values = {
            "DATE": "",
            "LOCATION": "",
            "PLACE": "",
            "RESTAURANT": "",
        }
        self.template = pd.read_csv(self.template_dir)
        self.nlu = NaturalLanguageUnderstanding()
        self.nlu.model_load()
        self.sbert = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        self.ood_answer_data = pd.read_csv('./dataset/ChatBotData.csv')
        self.ood_answer_data['embedding'] = self.ood_answer_data.apply(lambda row: self.sbert.encode(row.Q), axis=1)



    def cos_sim(self, A, B):
        return dot(A, B) / (norm(A) * norm(B))

    def return_ood_answer(self, question):
        embedding = self.sbert.encode(question)
        self.ood_answer_data['score'] = self.ood_answer_data.apply(lambda x: self.cos_sim(x['embedding'], embedding), axis=1)
        return self.ood_answer_data.loc[self.ood_answer_data['score'].idxmax()]['A']

    def make_search_key(self, nlu_result):
        intent = nlu_result.get("INTENT")
        keys = set()
        for name_value in nlu_result.get("SLOT"):
            slot_name = name_value.split("^")[0]
            slot_value = name_value.split("^")[1]
            keys.add(slot_name)
            self.values[slot_name] = slot_value
        slots = "^".join(keys)
        return intent, slots

    def search_template(self, nlu_result):
        intent, slots = self.make_search_key(nlu_result)
        matched_template = []

        for data in self.template.iterrows():
            intent_flag = False
            slot_flag = False

            row = data[1]
            if row['label'] == intent:
                intent_flag = True

            if isinstance(row.get("slot"), str):
                template_slots = sorted(row["slot"].split("^"))
                key_slots = sorted(slots.split("^"))
                if template_slots == key_slots:
                    slot_flag = True
            elif slots == "":
                slot_flag = True

            if intent_flag and slot_flag:
                matched_template.append(row["template"])

        return matched_template

    def replace_slot(self, flag, key, template):
        value = self.values.get(key)
        key = "{" + key + "}"
        if value != "":
            template = template.replace(key, value)
        else:
            template = ""
        flag = not flag
        return flag, template

    def filling_nlg_slot(self, templates):
        filling_templates = []

        for template in templates:
            date_index = template.find("{DATE}")
            location_index = template.find("{LOCATION}")
            place_index = template.find("{PLACE}")
            restraurant_index = template.find("{RESTRAURANT}")

            date_flag = date_index == -1
            location_flag = location_index == -1
            place_flag = place_index == -1
            restraurant_flag = restraurant_index == -1

            cnt = 0
            while (not (date_flag and location_flag and place_flag and restraurant_flag)):
                if not date_flag:
                    key = "DATE"
                    date_flag, template = self.replace_slot(date_flag, key, template)
                if not location_flag:
                    key = "LOCATION"
                    location_flag, template = self.replace_slot(location_flag, key, template)
                if not place_flag:
                    key = "PLACE"
                    place_flag, template = self.replace_slot(place_flag, key, template)
                if not restraurant_flag:
                    key = "RESTRAURANT"
                    restraurant_flag, template = self.replace_slot(restraurant_flag, key, template)

                filling_templates.append(template)

        return filling_templates

    def run_nlg(self, text):
        intent, predict = self.nlu.predict(text)
        nlu_result = self.nlu.convert_nlu_result(text, intent, predict)
        print("-------------------------------------------")
        print(nlu_result)
        print("-------------------------------------------")
        if nlu_result["INTENT"] == 'ood':
            result = self.return_ood_answer(text)
        else:
            templates = self.search_template(nlu_result)
            result = self.filling_nlg_slot(templates)

        return result, intent

    def run_nlg_crawl(self, text):
        crawler = SearchCrawler(self.nlu)
        nlg_result, intent = self.run_nlg(text)
        
        if intent=="ood" :
            return nlg_result
        
        crawler_result = crawler.run(text)
        print(crawler_result)
        result = nlg_result[0].replace('{CRAWL}', crawler_result)

        return result