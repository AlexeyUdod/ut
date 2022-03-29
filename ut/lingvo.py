import pymorphy2
import re
import numpy as np
import torch as tr
from ipymarkup import show_dep_markup
from wiki_ru_wordnet import WikiWordnet
from ruwordnet import RuWordNet

wwn = WikiWordnet()
wn = RuWordNet()
rwn = RuWordNet()
morph = pymorphy2.MorphAnalyzer()

t = tr.tensor
coo = tr.sparse_coo_tensor


cases_help = """Номинатив(Именительный падеж  Кто? Что? Подлежащее)
Генитив(Родительный падеж Кого? Чего? Принадлежность, состав, участие, происхождение, определение, отрицание)
Посессив(Притяжательный падеж Чей? Только принадлежность)
Датив(Дательный падеж Кому? Чему? Объект передачи, адресат речи, экспериенцер)
Аккузатив(Винительный падеж Кого? Что? Объект действия)
Эргатив(Действенный падеж Кто? Что? Субъект действия)
Абсолютив(Именительный падеж Кто? Кого? Что? Объект действия или состояния)
Аффектив(Дательный падеж. Кто? Кому? Субъект, воспринимающий что-либо или испытывающий какие-либо чувства)
Комитатив или Социатив(Совместный падеж С кем? Второстепенные субъекты действия)
Инструменталис(Творительный падеж Кем? Чем? Орудие действия; иногда субъект действия)
Партитив(Частичный падеж Чего? Действие переходит только на часть объекта)
Вокатив(Звательный падеж Обращение)
Эссив(Какой? Как? Нахождение в каком-либо состоянии)
Транслатив(Превратительный падеж Во что? Кем/чем стал?Изменение состояния или местонахождения)"""


def word2norm(word: str) -> str:
    """Return word normal form"""

    return morph.parse(word)[0].normal_form


def words2norm(words: list) -> list:
    """Return normal forms of word list"""

    return [word2norm(word) for word in words]


def tags(p: str, as_set = False) -> list:
    """Return morphological tags of input word"""

    if type(p) == str:
        p = morph.parse(p)[0]
    # else:
    #     p = [p]

    res = [[p.tag.POS,  # Part of Speech, часть речи
            p.tag.animacy,  # одушевленность
            p.tag.aspect,  # вид: совершенный или несовершенный
            p.tag.case,  # падеж
            p.tag.gender,  # род (мужской, женский, средний)
            p.tag.involvement,  # включенность говорящего в действие
            p.tag.mood,  # наклонение (повелительное, изъявительное)
            p.tag.number,  # число (единственное, множественное)
            p.tag.person,  # лицо (1, 2, 3)
            p.tag.tense,  # время (настоящее, прошедшее, будущее)
            p.tag.transitivity,  # переходность (переходный, непереходный)
            p.tag.voice] for p in p]

    if as_set:
        return [set(r) - {None} for r in res]
    else:
        return res[0]


def text2tags(text: str, ohe:'bool' = False) -> np.array:
    """Return array of words morphological tags for input text"""

    text = re.findall(r'\w+', text.lower())
    res = []
    for word in text:
        word = morph.parse(word)[0]
        res.append([word.normal_form] + tags(word))
    res = np.array(res)
    return res.T


def text4matrix(text: str)-> 'np.array[len()]':
    """Prepare human readable text matrix"""

    text2 = re.sub(r'[.,]', '', text.lower()).split(' ')

    shape = len(text2) + 1
    m = np.zeros((shape, shape), dtype='<U14')
    head = np.array([''] + text2)
    m[0] = head
    m[:,0] = head
    return m


def get_ind(m: 'np.array', x: str) -> int:
    """Get index of word for text martix"""

    ind = (m[0] == x).nonzero()[0][0]
    return ind


def triplets2matrix(m: 'np.array', triplets: 'np.array[..., 3]'):
    """Add triples to text matrix"""

    for s, p, o in triplets:
        m[get_ind(m, s), get_ind(m, o)] = p
    return m


#  http://opencorpora.org/dict.php?act=gram
tags_text = """1	POST	ЧР	часть речи	—
2	NOUN	СУЩ	имя существительное	POST
3	ADJF	ПРИЛ	имя прилагательное (полное)	POST
4	ADJS	КР_ПРИЛ	имя прилагательное (краткое)	POST
5	COMP	КОМП	компаратив	POST
6	VERB	ГЛ	глагол (личная форма)	POST
7	INFN	ИНФ	глагол (инфинитив)	POST
8	PRTF	ПРИЧ	причастие (полное)	POST
10	PRTS	КР_ПРИЧ	причастие (краткое)	POST
11	GRND	ДЕЕПР	деепричастие	POST
12	NUMR	ЧИСЛ	числительное	POST
13	ADVB	Н	наречие	POST
14	NPRO	МС	местоимение-существительное	POST
15	PRED	ПРЕДК	предикатив	POST
16	PREP	ПР	предлог	POST
17	CONJ	СОЮЗ	союз	POST
18	PRCL	ЧАСТ	частица	POST
19	INTJ	МЕЖД	междометие	POST
21	ANim	Од-неод	категория одушевлённости	—
22	anim	од	одушевлённое	ANim
23	inan	неод	неодушевлённое	ANim
24	GNdr	хр	род / род не выражен	—
25	masc	мр	мужской род	ms-f
26	femn	жр	женский род	ms-f
27	neut	ср	средний род	GNdr
28	ms-f	мж	общий род (м/ж)	GNdr
29	NMbr	Число	число	—
30	sing	ед	единственное число	NMbr
31	plur	мн	множественное число	NMbr
32	Sgtm	sg	singularia tantum	—
33	Pltm	pl	pluralia tantum	—
36	Fixd	0	неизменяемое	—
37	CAse	Падеж	категория падежа	—
38	nomn	им	именительный падеж	CAse
39	gent	рд	родительный падеж	CAse
40	datv	дт	дательный падеж	CAse
41	accs	вн	винительный падеж	CAse
42	ablt	тв	творительный падеж	CAse
43	loct	пр	предложный падеж	CAse
44	voct	зв	звательный падеж	nomn
45	gen1	рд1	первый родительный падеж	gent
46	gen2	рд2	второй родительный (частичный) падеж	gent
47	acc2	вн2	второй винительный падеж	accs
48	loc1	пр1	первый предложный падеж	loct
49	loc2	пр2	второй предложный (местный) падеж	loct
50	Abbr	аббр	аббревиатура	—
51	Name	имя	имя	—
52	Surn	фам	фамилия	—
53	Patr	отч	отчество	—
54	Geox	гео	топоним	—
55	Orgn	орг	организация	—
56	Trad	tm	торговая марка	—
57	Subx	субст?	возможна субстантивация	—
58	Supr	превосх	превосходная степень	—
59	Qual	кач	качественное	—
60	Apro	мест-п	местоименное	—
61	Anum	числ-п	порядковое	—
62	Poss	притяж	притяжательное	—
63	V-ey	*ею	форма на -ею	—
64	V-oy	*ою	форма на -ою	—
65	Cmp2	сравн2	сравнительная степень на по-	—
66	V-ej	*ей	форма компаратива на -ей	—
67	ASpc	Вид	категория вида	—
68	perf	сов	совершенный вид	ASpc
69	impf	несов	несовершенный вид	ASpc
70	TRns	Перех	категория переходности	—
71	tran	перех	переходный	TRns
72	intr	неперех	непереходный	TRns
73	Impe	безл	безличный	—
74	Impx	безл?	возможно безличное употребление	—
75	Mult	мног	многократный	—
76	Refl	возвр	возвратный	—
77	PErs	Лицо	категория лица	—
78	1per	1л	1 лицо	PErs
79	2per	2л	2 лицо	PErs
80	3per	3л	3 лицо	PErs
81	TEns	Время	категория времени	—
82	pres	наст	настоящее время	TEns
83	past	прош	прошедшее время	TEns
84	futr	буд	будущее время	TEns
85	MOod	Накл	категория наклонения	—
86	indc	изъяв	изъявительное наклонение	MOod
87	impr	повел	повелительное наклонение	MOod
88	INvl	Совм	категория совместности	—
89	incl	вкл	говорящий включён (идем, идемте)	INvl
90	excl	выкл	говорящий не включён в действие (иди, идите)	INvl
91	VOic	Залог	категория залога	—
92	actv	действ	действительный залог	VOic
93	pssv	страд	страдательный залог	VOic
94	Infr	разг	разговорное	—
95	Slng	жарг	жаргонное	—
96	Arch	арх	устаревшее	—
97	Litr	лит	литературный вариант	—
98	Erro	опеч	опечатка	—
99	Dist	искаж	искажение	—
100	Ques	вопр	вопросительное	—
101	Dmns	указ	указательное	—
103	Prnt	вводн	вводное слово	—
104	V-be	*ье	форма на -ье	—
105	V-en	*енен	форма на -енен	—
106	V-ie	*ие	форма на -и- (веселие, твердостию); отчество с -ие	—
107	V-bi	*ьи	форма на -ьи	—
108	Fimp	*несов	деепричастие от глагола несовершенного вида	—
109	Prdx	предк?	может выступать в роли предикатива	—
110	Coun	счетн	счётная форма	—
111	Coll	собир	собирательное числительное	—
112	V-sh	*ши	деепричастие на -ши	—
113	Af-p	*предл	форма после предлога	—
114	Inmx	не/одуш?	может использоваться как одуш. / неодуш.	—
115	Vpre	в_предл	Вариант предлога ( со, подо, ...)	—
116	Anph	анаф	Анафорическое (местоимение)	—
117	Init	иниц	Инициал	—
118	Adjx	прил?	может выступать в роли прилагательного	—
119	Ms-f	ор	колебание по роду (м/ж/с): кофе, вольво	—
120	Hypo	гипот	гипотетическая форма слова (победю, асфальтовее)	—"""

tags_text2 = """LATN	Токен состоит из латинских букв (например, “foo-bar” или “Maßstab”)
PNCT	Пунктуация (например, , или !? или …)
NUMB	Число (например, “204” или “3.14”)
intg	целое число (например, “204”)
real	вещественное число (например, “3.14”)
ROMN	Римское число (например, XI)
UNKN	Токен не удалось разобрать"""

tags_list = np.array([w.split('\t') for w in tags_text.split('\n')])
tags_list2 = np.array([w.split('\t') for w in tags_text2.split('\n')])

tags_dic = {x[1]:{'num':int(x[0]), 'id2':x[2], 'info':x[3], 'hyper':x[4]} for x in tags_list}
tags_dic.update({x[0]:{'num':i + 121, 'info':x[1]} for i, x in enumerate(tags_list2)})
tags_dic.update({tags_dic[k]['num']:k for k in tags_dic.keys()})


def tags_n(word: str)->'list of tags numbers':
    """Return list of word graphem tags codes"""

    return [tags_dic[tag]['num'] for tag in tags(word) if tag is not None]


def get_level_tags(tag: 'str tag') -> 'list of level tags':
    """Get list of same level graphem tag in utl.tags_dic"""

    try:
        hyper = tags_dic[tag]['hyper']
        level_tags = [x for x in tags_dic if 'hyper' in tags_dic[x] and tags_dic[x]['hyper'] == hyper]
        return level_tags
    except:
        return None


stop = {'punct', 'root'}
swap_dic = {'nsubj': 'что делал', 'nsubj:pass': 'что с ним делали', 'parataxis':'связь>>',}
rel_dic = {'amod':'какой', 'det':'какой', 'iobj': 'кому', 'obj': 'что',
           'nmod':'чего', 'case': 'предл', 'obl':'как', 'conj': 'связь+', 'cc':'союз+',
           'appos': 'по имени', 'advmod': 'как', 'flat:name': 'имя', 'flat:foreign':'ино.',
           'ccomp': 'как', 'cop': 'было', 'fixed':'выраж', 'cop': 'было', 'fixed':'выраж',
           'xcomp': 'чем', 'mark':'союз>>', 'acl': 'что делающей', 'nummod:gov': 'числ',
           'orphan': 'пропуск', 'advcl':'связь>>',  'nummod':'числ', 'obl:agent':'кем',
           'discourse':'как',}

primitives = """Полный набор кандидатов на роль универсальных семантических примитивов,
 возникший в результате более чем двух десятилетий эмпирических поисков и сопоставлений, 
 включает следующие элементы:

[субстантивы]
я, ты, кто-то, что-то, люди

[детерминаторы, квантификаторы]
этот, тот же самый, другой, один, два, все/весь, много

[предикаты ментальных состояний]
знать, хотеть, думать, говорить, чувствовать

[действия, события]
делать, происходить/случаться

[оценка]
хороший, плохой

[дескрипторы]
большой, маленький

[интенсификатор]
очень

[метапредикаты]
не/нет (отрицание), если, из-за, мочь, очень, подобный/как

[время и место]
когда, где, после (до), под (над)

[таксономия, партономия]
вид/разновидность, часть"""


def word2def(word: 'str') -> 'list of str':
    """Get word definitions from WikiWordNet lib"""

    res = []
    for i1, syn in enumerate(wwn.get_synsets(word)):
        for i2, w in enumerate(syn.get_words()):
            res.append(w.definition())
    return res


def word2syn_rwn(word:'str'
                ) -> 'WikiWordnet.synset':
    """Get word synsets"""

    res = wn.get_synsets(word)
    return res


def main_words(m: 'np.array: sentense adj matrix',
               lemmas: 'list (or np.array) of sentense words'=None
               ) -> 'np.array of main words in sentense':
    """Extract main word (with output links only) of sentense"""

    if lemmas is None:
        lemmas = m[0,1:]
    deg_in = tr.tensor(m != '').sum(dim=0)[1:] - 1
    deg_out = tr.tensor(m != '').sum(dim=1)[1:] - 1
    inds = (((deg_in == 0) * (deg_out > 0)).nonzero(as_tuple=True)[0]).tolist()
    res = np.array(lemmas)[inds]
    return res


# word seqs

def w2s(word):
    """Convert word to letters sequences"""
    
    seqs = []
    for i, l1 in enumerate(word):
        for l2 in word[i+1:]:
            
            seqs.append(l1 + l2)
    return set(seqs)


def ws2ss(words_list):
    """Convert list of words to list of letters sequences"""
    
    seqs = []
    for word in words_list:
        seqs.append(w2s(word))
    return seqs



def swap_list2dic(swap):
    """Turn swap list of simular letters to swap dic"""
    
    swap_dic = {}
    for ks in swap:
        for k in ks:
            swap_dic[k] = ks

    return swap_dic


def swap_letters(word, swap_dic):
    """Swap letters in word to swap_dic vals"""
    
    lword = list(word)
    for i in range(len(lword)):
        if lword[i] in swap_dic:
            lword.insert(i, swap_dic[lword[i]])
            lword.pop(i+1)
    return ''.join(lword)


def simp(word1, word2, swap = None):
    """Return probabilites of sequental implication 
    word1 in word2 and vice versa"""
    
    seq1 = w2s(word1)
    
    if swap is not None:
        if swap:
            swap = ['fvbpфвбп', 'jcxgkqhцчгкх', 'dtдт', 
                    'srzсршжз','lл', 'oоuув','aа','eе' , 'iи']
        if type(swap) is list:
            swap = swap_list2dic(swap)
        word2 = swap_letters(word2, swap)
    
    seq2 = w2s(word2)
    intr = len(seq1 & seq2) 
    imp1 = intr / len(seq1) 
    imp2 = intr / len(seq2) 
    return imp1