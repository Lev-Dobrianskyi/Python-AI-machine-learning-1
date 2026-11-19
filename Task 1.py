from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "Жахливе обслуговування, офіціант був грубий і довелося чекати пів години.",
    "Кава була холодна, а тістечко черстве. Більше не прийду.",
    "Дуже брудно, столи брудні, а в туалеті жах.",
    "Їжа несмачна і пересолена, а ціни завищені.",
    "Найгірший заклад у місті, обслуговування повільне і персонал непривітний.",
    "Замовлення принесли холодним і зіпсованим, гроші повертати відмовилися.",
    "Брудна підлога, брудний посуд і неприємний запах у залі.",
    "Кава гірка і несмачна, а круасан сухий і черствий.",
    "Довге очікування, грубий персонал і жахлива атмосфера.",
    "Дорого і несмачно, порції маленькі, а якість жахлива.",
    "Відвідування було розчаруванням: брудно, шумно і повільно.",
    "Обслуговування жахливе, їжа гидка, більше ніколи.",

    "Чудова кава та дуже привітний персонал, обов'язково повернуся!",
    "Смачна їжа, затишна атмосфера і швидке обслуговування.",
    "Найкращі круасани в місті, все свіже та дуже смачне.",
    "Приємний заклад, чисто, затишно і недорого.",
    "Обслуговування швидке, офіціанти ввічливі, кава чудова.",
    "Люблю це місце: смачні десерти і приємна музика.",
    "Все було дуже смачно, порції великі, ціни прийнятні.",
    "Затишна кав'ярня з чудовим персоналом і свіжою випічкою.",
    "Чудове місце для сніданку, все свіже і дуже смачне.",
    "Швидко, смачно і привітно — рекомендую всім!",
    "Приємна атмосфера, чисто і дуже смачна кава.",
    "Найкращий заклад у районі, обслуговування на вищому рівні.",
]

labels = [1,1,1,1,1,1,1,1,1,1,1,1,
          0,0,0,0,0,0,0,0,0,0,0,0]

var_vex = CountVectorizer()
x = var_vex.fit_transform(texts)
model = MultinomialNB().fit(x, labels)

tests = [
    "Кава була холодна, а обслуговування повільне і грубе.",
    "Дуже брудно і несмачно, ціни завищені.",
    "Жахлива їжа, довелося довго чекати, більше не прийду.",
    "Персонал непривітний, а тістечко черстве.",
    "Найгірший заклад, все гидке і дороге.", # gives a false negative using ml method

    "Смачна кава і дуже привітний персонал, рекомендую!",
    "Затишно, чисто і швидке обслуговування.",
    "Чудові десерти та приємна атмосфера, все сподобалось.",
    "Свіжа випічка і ввічливі офіціанти.",
    "Найкращі круасани в місті, дуже смачно!",
]

predictions = model.predict(var_vex.transform(tests))
for text, pred in zip(tests, predictions):
    label = "NEGATIVE" if pred == 1 else "POSITIVE"
    print(f"[{label}] {text}")

#//////////////////////////////////////////////////////////// without ml

def negative_rule(line):
    line = line.lower()
    negative_word = [
        "trash", "bad", "garbage", "slow", "unpleasant", "stink", "dirty",
        "worst", "expensive", "disgusting", "horrible", "terrible", "awful",
        "poor", "rude", "mediocre", "overpriced", "cold", "stale", "bland",
        "greasy", "crowded", "noisy", "disappointing", "unfriendly", "filthy",
        "messy", "broken", "inedible", "nasty", "soggy", "burnt", "wait",
        "long wait", "never again", "not worth", "waste", "rip-off", "ripoff"
    ]
    return any(word in line for word in negative_word)


tests = [
    "This cafe is pretty bad, I wouldn't recommend it because of how expensive it is to get a small serving",
    "Slow time of arrival, I noticed the place was dirty too",
    "The food was absolutely terrible and the service was rude",
    "Amazing experience, would definitely come back!",
    "The staff were unfriendly and the place was crowded and noisy",
    "Best coffee I've ever had, the pastries were fresh",
    "The meal was cold and bland, total waste of money",
    "The waiter was polite and the atmosphere was lovely", # gives a false positive using bad_words method
    "Overpriced and disappointing, never again",
    "The bread was stale and the fries were soggy",
    "A wonderful little spot with great ambiance",
    "The bathroom was filthy and the tables were messy",
    "The chicken was burnt and completely inedible",
    "Lovely staff, quick service, and delicious food"
]

for test in tests:
    if negative_rule(test):
        print(f"Negative comment: {test}")
    else:
        print(f"Not a negative comment: {test}")