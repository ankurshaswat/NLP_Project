import pickle

from wikidata.client import Client
from tqdm import tqdm
# import json

# from SPARQLWrapper import SPARQLWrapper, JSON

all_data = {}
try:
    all_data = pickle.load(open("Wiki/save.pkl", "rb"))
except:
    all_data = {}

client = Client()


def decode_list(list_of_codes):
    decoded_list = []
    for code in tqdm(list_of_codes):
        if code not in all_data:
            name = code
            if code[-4:] == '_inv':
                name = code[:-4]
            try:
                entity = client.get(name, load=True)
                all_data[code] = {
                    'code': code, 'label': entity.label, 'description': entity.description}
            except:
                all_data[code] = {
                    'code': code, 'label': 'NOT FOUND', 'description': 'NOT FOUND'}
        decoded_list.append(all_data[code])
    pickle.dump(all_data, open("Wiki/save.pkl", "wb"))
    return decoded_list


if __name__ == '__main__':
    print(decode_list(['P1', 'P2667']))
# def get_results(endpoint_url, query):
#     sparql = SPARQLWrapper(endpoint_url)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     return sparql.query().convert()
# data = json.load(open('Wiki/symbol2ids'))
# endpoint_url = "https://query.wikidata.org/sparql"


# i = 0

# num = len(data.keys())
# print(num)

# query_list = ''
# my_list = []

# for code in data.keys():
#     if(i%100 == 99):
#         query = """SELECT ?itemLabel ?itemDescription WHERE {
#         VALUES ?item {""" + '\n' + query_list + """}
#         SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
#         }"""
#         # print(query)
#         results = get_results(endpoint_url, query)
#         print(results)
#         j = 0
#         for result in results["results"]["bindings"]:
#             all_data[my_list[j]] = {'label':result["itemLabel"]["value"],'value':my_list[j],'description':result["itemDescription"]["value"]}
#             print(my_list[j],result["itemLabel"]["value"])
#             j += 1
#         pickle.dump( all_data, open( "Wiki/save.pkl", "wb" ) )
#         i = 0
#         query_list = ''
#         my_list = []

#     if code in all_data:
#         continue

#     if code[0] =='P':
#         continue

#     if code[-4:] == '_inv':
#         if code[:-4] in query_list:
#             continue
#         query_list += 'wd:' + code[:-4] + '\n'
#         my_list.append(code)
#         # entity = client.get(code[:-4],load=False)
#     else:
#         if code in query_list:
#             continue
#         query_list += 'wd:' + code + '\n'
#         my_list.append(code)

#         # entity = client.get(code,load=False)

#     # all_data[code] = entity.label
#     # print(entity.label)
#     # print(i)

#     i += 1

# # pip install sparqlwrapper
# # https://rdflib.github.io/sparqlwrapper/
# # pip install sparqlwrapper
# # https://rdflib.github.io/sparqlwrapper/

# print('done')


# # print(query)
# # print(results)

# # for result in results:
# #     print(result['results'])
# # for result in results["results"]["bindings"]:
# #     print(result)
