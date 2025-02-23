import pickle

with open("feat_map.pkl", 'rb') as fi:
    feat_map = pickle.load(fi)
with open("defaults.pkl", 'rb') as fi:
    defaults = pickle.load(fi)
with open("offset.pkl", 'rb') as fi:
    field_offset = pickle.load(fi)

print(feat_map)
print(defaults)
print(field_offset)