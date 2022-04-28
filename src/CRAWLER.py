import requests
from bs4 import BeautifulSoup

from src.NLU import NaturalLanguageUnderstanding


class SearchCrawler:
    def __init__(self, nlu):
        self.url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='
        self.nlu = nlu


    def get_dust_search_result(self, entity_search_keyword):
        req = requests.get(self.url + entity_search_keyword)
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')

        dust = soup.select('.cs_air_environment .detail_info dl')

        if '모레' in entity_search_keyword:
            dust_am = dust[4].get_text()
            dust_pm = dust[5].get_text()
            dust_am = dust_am.split(' ')[1] + ' ' + dust_am.split(' ')[2]
            dust_pm = dust_pm.split(' ')[1] + ' ' + dust_pm.split(' ')[2]
            return dust_am + ' ' + dust_pm

        if '내일' in entity_search_keyword:
            dust_am = dust[2].get_text()
            dust_pm = dust[3].get_text()
            dust_am = dust_am.split(' ')[1] + ' ' + dust_am.split(' ')[2]
            dust_pm = dust_pm.split(' ')[1] + ' ' + dust_pm.split(' ')[2]
            return dust_am + ' ' + dust_pm

        dust_am = dust[0].get_text()
        dust_pm = dust[1].get_text()
        dust_am = dust_am.split(' ')[1] + ' ' + dust_am.split(' ')[2]
        dust_pm = dust_pm.split(' ')[1] + ' ' + dust_pm.split(' ')[2]
        return dust_am + ' ' + dust_pm

    def get_weather_search_result(self, entity_search_keyword):
        req = requests.get(self.url + entity_search_keyword)
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')

        low_temperature = soup.select('.cs_weather_new .weekly_forecast_area .week_item .cell_temperature .lowest')
        high_temperature = soup.select('.cs_weather_new .weekly_forecast_area .week_item .cell_temperature .highest')

        if '모레' in entity_search_keyword:
            low_temperature = low_temperature[2]
            high_temperature = high_temperature[2]
            crawl_result = low_temperature.get_text() + ' ' + high_temperature.get_text()
            return crawl_result

        if '내일' in entity_search_keyword:
            low_temperature = low_temperature[1]
            high_temperature = high_temperature[1]
            crawl_result = low_temperature.get_text() + ' ' + high_temperature.get_text()
            return crawl_result

        low_temperature = low_temperature[0]
        high_temperature = high_temperature[0]
        crawl_result = low_temperature.get_text() + ' ' + high_temperature.get_text()

        return crawl_result

    def make_search_keyword(self, nlu_result):
        intent = nlu_result['INTENT']

        if intent == 'dust':
            intent = '미세먼지'
        elif intent == 'restaurant':
            intent = '맛집'
        elif intent == 'travel':
            intent = '여행'
        elif intent == 'weather':
            intent = '날씨'

        entitys = []
        for entity in nlu_result['SLOT']:
            entitys.append(entity.split('^')[1])

        entity_search_keyword = ' '.join(entuty for entuty in entitys)
        return entity_search_keyword + ' ' + intent

    def run(self, text):
        intent, predict = self.nlu.predict(text)
        nlu_result = self.nlu.convert_nlu_result(text, intent, predict)
        entity_search_keyword = self.make_search_keyword(nlu_result)

        if intent == 'dust':
            crawl_result = self.get_dust_search_result(entity_search_keyword)
        elif intent == 'weather':
            crawl_result = self.get_weather_search_result(entity_search_keyword)

        return crawl_result
