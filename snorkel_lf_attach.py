from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from enumerate_biomimicry_functions import enumerate_functions
import pandas as pd

ABSTAIN = -1

#Questions:
#How do automatically enumerate the functions in a way snorkel can recognize?
#Best way to write label functions?
attach_permanently = 0
attach_temporarily = 1

enum_bio = pd.read_csv("./biomimicry_functions_enumerated.csv")
# enum_bio["function"] = enum_bio["function"].apply(clean_text)
enum_bio = enum_bio.set_index("function")

# Load phrase heuristics
function_rule_phrases = pd.read_csv("./biomimicry_function_rules.csv")
functions = ['attach_temporarily', 'attach_permanently'] #do we want to just pull the rows where 'Attach' is in level1?
lfs_attach = []

#create lists of rules for each attach function and add underscores
#should this be in a separate file?
keyword_lst = []
def create_keyword_list(functions):
    for label in functions:
        lst = function_rule_phrases[f'{label}_rules'].to_list()
        newlist = [x for x in lst if pd.isnull(x) == False]
    for rule in newlist:
        rule = rule.replace(" ", "_")
        keyword_lst.append(rule)
    return keyword_lst

create_keyword_list(function_rule_phrases)

#create keyword lfs for every rule in each attach function
class CreateKeywordLfs:

    def keyword_lookup(self, x, keywords, label):
        """[summary]

        Args:
            x ([type]): [description]
            keywords ([type]): [description]
            label ([type]): [description]

        Returns:
            [type]: [description]
        """
        for label in functions: #is this correct? can label be in '' and do we use label argument to iterate through functions list? 
            if any(word in x.text.lower() for word in keywords):
                return label
            return ABSTAIN

#should we use the LF David wrote instead, that enumerates each label and creates enumerated list?
# def keyword_lookup(x, keywords, label):
#     if any(word in x.title.lower() or x.abstract.lower() for word in keywords):
#         return enum_bio.loc[label, "function_enumerated"] 
#     return ABSTAIN


    def make_keyword_lf(self, keywords, label=attach_permanently): #how can we automate the 'label' argument?
        for label in functions:
            return LabelingFunction(
                name=f"keyword_{keywords}",
                f=self.keyword_lookup,
                resources=dict(keywords=keywords, label=label),
            )
        
    index = 0
    for index, keyword in enumerate(functions):
        make_keyword_lf(keywords=functions[index], label=keyword)
        index += 1


#create list of lfs for attach functions
    def get_attach_lfs(self, functions):
        for label in functions: #can we use 'label' here?
            name = f"keyword_{label}"
            lfs_attach.append(name)
        return lfs_attach




# keyword_attach_firmly = make_keyword_lf(keywords=["attach firmly"])
# keyword_attaching_firmly = make_keyword_lf(keywords=["attaching firmly"])

# """Spam comments talk about 'my channel', 'my video', etc."""
# keyword_my = make_keyword_lf(keywords=["my"])

# """Spam comments ask users to subscribe to their channels."""
# keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

# """Spam comments post links to other channels."""
# keyword_link = make_keyword_lf(keywords=["http"])

# """Spam comments make requests rather than commenting."""
# keyword_please = make_keyword_lf(keywords=["please", "plz"])

# """Ham comments actually talk about the video's content."""
# keyword_song = make_keyword_lf(keywords=["song"], label=HAM)

# @labeling_function()
# def regex_check_out(x):
#     return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

# lfs = [
#     keyword_my,
#     keyword_subscribe,
#     keyword_link,
#     keyword_please]

# def create_keyword_list(*argv):
#     for arg in argv:
#         lst = function_rule_phrases[f'{arg}_rules'].to_list()
#     for rule in lst:
#         rule = rule.replace(" ", "_")
#         lst.append(rule)
#     return lst

# #create keyword list for attach_permanently and attach_temporarily
# keywords_attach_permanently = function_rule_phrases['attach_permanently_rules'].to_list()
# keywords_attach_temporarily = function_rule_phrases['attach_temporarily_rules'].to_list()

# #apply underscore to all rules in keyword lists
# keywords_attach_permanently_underscore = []
# for keyword in keywords_attach_permanently:
#     keyword = keyword.replace(" ", "_")
#     keywords_attach_permanently_underscore.append(keyword)

# keywords_attach_temporarily_underscore = []
# for keyword in keywords_attach_temporarily:
#     keyword = keyword.replace(" ", "_")
#     keywords_attach_temporarily_underscore.append(keyword)

#create keyword lfs for attach_temporarily
# def keyword_lookup(x, keywords, label):
#     if any(word in x.text.lower() for word in keywords):
#         return label
#     return ABSTAIN

# def make_keyword_lf(keywords, label=enumerate_functions('attach_temporarily')):
#     return LabelingFunction(
#         name=f"keyword_{keywords}",
#         f=keyword_lookup,
#         resources=dict(keywords=keywords, label=label),
#     )
    
# index = 0
# for index, keyword in enumerate(keywords_attach_temporarily_underscore):
#      make_keyword_lf(keywords=keywords_attach_temporarily_underscore[index], label=enumerate_functions('attach_temporarily'))
#      index += 1