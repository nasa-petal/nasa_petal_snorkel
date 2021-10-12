import itertools
from snorkel_lf_attach import CreateKeywordLfs 
from snorkel_lf_move import get_move_lfs



attach = CreateKeywordLfs.get_attach_lfs()
move = get_move_lfs()

lfs = list(itertools.chain(attach, move))

print(lfs)

# def get_lfs():
#     for function in dir(snorkel_lf_attach):
#         if "lf_" in function:
#             # lfs.append(getattr(snorkel_lf_attach, function))
#             lfs.append(function[3:])
#         elif "keyword_" in function:
#             # lfs.append(getattr(snorkel_lf_attach, function))
#             lfs.append(function)
#     lfs.remove('make_keyword_lf')
#     return lfs


# for function in dir(snorkel_lf_attach):
#     if "lf_" in function:
#         # lfs.append(getattr(snorkel_lf_attach, function))
#         lfs.append(function[3:])
#     elif "keyword_" in function:
#         # lfs.append(getattr(snorkel_lf_attach, function))
#         lfs.append(function)
# # lfs.remove('make_keyword_lf')
# # lfs.remove('keyword_lookup')
# print(lfs)