from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from word2number import w2n
import pandas as pd

class Order_Parser():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("davanstrien/deberta-v3-base_fine_tuned_food_ner")
        model = AutoModelForTokenClassification.from_pretrained("davanstrien/deberta-v3-base_fine_tuned_food_ner")
        self.pipe = pipeline("ner", model=model, tokenizer=tokenizer)
        self.complete_order_dict={}

    def restart_state(self):
        self.complete_order_dict={}

    def join_adjacent_items(self, data):
        result = []
        current_group = []
        current_entity = None

        for item in data:
            # Check if the item's entity is related to FOOD or QUANTITY
            if any(e in item['entity'] for e in ['FOOD', 'QUANTITY']):
                # Start a new group if the entity type changes
                if not current_entity:
                    current_entity = item['entity'].split('-')[-1]
                elif current_entity != item['entity'].split('-')[-1]:
                    result.append({'entity': current_entity, 'word': ''.join(current_group)})
                    current_group = []
                    current_entity = item['entity'].split('-')[-1]
                    
                current_group.append(item['word'])
            else:
                if current_group:
                    result.append({'entity': current_entity, 'word': ''.join(current_group)})
                    current_group = []
                    current_entity = None
                result.append(item)

        # Handle the last group if it exists
        if current_group:
            result.append({'entity': current_entity, 'word': ''.join(current_group)})

        return result
    
    def order_parser(self, sentence):
        sentence = sentence.replace(',', ' ')
        sentence = sentence.replace('?', ' ')
        sentence = sentence.replace('.', ' ')
        # updated_sentence = updated_sentence.replace('and', 'one')
        sentence = sentence.replace(' a ', ' one ')
        sentence = sentence.replace(' an ', ' one ')
        # sentence = sentence.replace(' and ', ' one ')
        # print(updated_sentence)
        # raw_order = self.pipe(updated_sentence)
        print(sentence)
        raw_order = self.pipe(sentence)
        raw_order_piped = self.join_adjacent_items(raw_order)
        
        order_dict={}
        quantity_exist = False
        for ent in raw_order_piped:
            if 'QUANTITY' in ent['entity']:
                quantity = ent['word'].replace('▁', ' ')
                try:
                    quantity = w2n.word_to_num(quantity)
                except:
                    quantity = 1

                # print(quantity)
                quantity_exist = True

            elif 'FOOD' in ent['entity']:
                food = ent['word'].replace('▁', '')
                # print(food)
                
                if quantity_exist:
                    order_dict[food] = quantity
                else:
                    order_dict[food] = 1

                quantity_exist=False

        # print(order_dict)
        self.complete_order_dict.update(order_dict)
        return self.complete_order_dict
    
