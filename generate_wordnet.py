"""Generate vocabulary.json from Princeton WordNet using NLTK.
Run locally: python -m pip install nltk
             python -m nltk.downloader wordnet omw-1.4
             python generate_wordnet.py
Keep this script outside the published website if desired.
"""
import json, re, pathlib, collections
from nltk.corpus import wordnet as wn
ROOT=pathlib.Path(__file__).resolve().parent
existing=json.loads((ROOT/'vocabulary.json').read_text())
# Retain original curated anchors by slicing at the original length recorded below.
ANCHORS={"opposition":97,"indirect":150,"remote":176,"far":120}
anchors={k:existing['pools'][k][:n] for k,n in ANCHORS.items()}
# Source text used for the first exclusion pass (KJV Genesis 11:1-9).
genesis="""And the whole earth was of one language and of one speech And it came to pass as they journeyed from the east that they found a plain in the land of Shinar and they dwelt there And they said one to another Go to let us make brick and burn them throughly And they had brick for stone and slime had they for morter And they said Go to let us build us a city and a tower whose top may reach unto heaven and let us make us a name lest we be scattered abroad upon the face of the whole earth And the LORD came down to see the city and the tower which the children of men builded And the LORD said Behold the people is one and they have all one language and this they begin to do and now nothing will be restrained from them which they have imagined to do Go to let us go down and there confound their language that they may not understand one another speech So the LORD scattered them abroad from thence upon the face of all the earth and they left off to build the city Therefore is the name of it called Babel because the LORD did there confound the language of all the earth and from thence did the LORD scatter them abroad upon the face of all the earth"""
excluded=set(re.findall('[a-z]+',genesis.lower()))
concepts='babel tower building architecture architect brick mortar stone construction city heaven god language speech tongue unity pride ambition ascent dispersal confusion monument'.split()
for concept in concepts:
 excluded.add(concept)
 for syn in wn.synsets(concept):
  for lemma in syn.lemmas():
   excluded.add(lemma.name().lower().replace('_',''))
  for near in syn.hyponyms()+syn.hypernyms():
   for lemma in near.lemmas(): excluded.add(lemma.name().lower().replace('_',''))
# WordNet path similarity is defined primarily for noun/verb taxonomies and is not
# a universal semantic distance. We use semantic domains as a transparent proxy.
domains={
 'indirect':['feeling','sensation','texture','motion','color','emotion','perception'],
 'remote':['food','fruit','flower','animal','bird','plant','fish','insect'],
 'far':['physics','chemistry','mathematics','music','instrument','biology','astronomy','computer']
}
word_to_domains=collections.defaultdict(set)
for category, seeds in domains.items():
 for seed in seeds:
  for syn in wn.synsets(seed, pos=wn.NOUN)[:4]:
   for node in syn.closure(lambda x:x.hyponyms(),depth=5):
    for lemma in node.lemmas():
     word_to_domains[lemma.name().lower().replace('_','')].add(category)
# Retain common readable single words; include all four POS in a supplemental pool.
allwords=set()
for syn in wn.all_synsets():
 for lemma in syn.lemmas():
  word=lemma.name().lower()
  if word.isascii() and word.isalpha() and 4<=len(word)<=14 and word not in excluded:
   allwords.add(word)
new={k:list(v) for k,v in anchors.items()}
for word in sorted(allwords):
 if word in excluded or word in set().union(*map(set,anchors.values())):continue
 matched=word_to_domains.get(word,set())
 if 'far' in matched:group='far'
 elif 'remote' in matched:group='remote'
 elif 'indirect' in matched:group='indirect'
 else:continue # no credible semantic grouping; omit rather than invent distance
 if len(new[group])<5000:new[group].append(word)
for key in new:new[key]=list(dict.fromkeys(new[key]))
output={'meta':{'source':'Princeton WordNet via NLTK, plus original BABEL curated anchors','method':'KJV Genesis 11:1-9 token exclusion, Babel seed synonym/taxonomy exclusions, WordNet domain-based grouping; manual review recommended','counts':{k:len(v) for k,v in new.items()}},'pools':new}
(ROOT/'vocabulary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print('Generated:',output['meta']['counts'])
