import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import math
from datetime import datetime
from scipy.stats import levene
import warnings



df = pd.read_csv('/datasets/games.csv')
display(df.head(10))
display(df.info())



# Ознакомимся со значениями в столбцах.

for row in df:
    print(row)
    print()
    print(df[row].value_counts())
    print(df[row].unique())
    print(len(df[row]))
    print('---' * 20)



df.columns = df.columns.str.lower()

columns = list(df)
for column in columns:
    if df[column].dtype == object:
        df[column] = df[column].str.lower()

display(df.head(10))


# Удалим пропуски в столбцах name year_of_release и genre. Т.к их суммарное количество равно 273, это меньше 2%.  
# Заменим тип данных в столбце year_of_release на целочисленный.  
# А в столбце user_score на float. Для этого значения tbd заменим на nan.


df.dropna(subset = ['name', 'year_of_release', 'genre'], inplace = True)
df['year_of_release'] = df['year_of_release'].astype('int')
df.loc[df['user_score'] == 'tbd', 'user_score'] = float('nan')
df['user_score'] = df['user_score'].astype('float')
print(df.info())



def sum_sales(df):
    return df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']
df['sum_sales'] = df.apply(sum_sales,axis = 1)
display(df)



all_sales_genre = df.pivot_table(index = 'genre',values = ['eu_sales','na_sales','jp_sales','other_sales'],aggfunc = 'sum')
#display(all_sales_genre)


# Из таблици видно, что в Северной Америке игры пользуются наибольшим спросом, продажи почти во всех жанрах выше, чем в других регионах. Только в Японии жанр role-playing опередил Северную Америку.

# Найдём количество выпущенных игр в разные годы.


games_in_years = df.pivot_table(index = 'year_of_release', values = 'sum_sales',aggfunc = 'count')
display(games_in_years)
games_in_years.plot(kind='bar', figsize=(15,5))
plt.ylabel('sum_sales')
plt.show()


platforms_sales = df.pivot_table(index = 'platform',values = 'sum_sales', aggfunc = 'sum')
display(platforms_sales.sort_values(by = 'sum_sales', ascending=False))
platforms_sales.plot(kind='bar', figsize=(15,5))
plt.ylabel('sum_sales')
plt.show()


years_of_life =[]
for game_platform in ['ps2','x360','ps3','wii','ds']:
    df[df['platform'] == game_platform].pivot_table(index='year_of_release', values='sum_sales', aggfunc='sum').plot(kind='bar', figsize=(15,5))
    plt.title(game_platform)
    plt.ylabel('sum_sales')
    plt.show()
    years_of_life.append(df.query(f'platform == "{game_platform}"')['year_of_release'].max()-df.query(f'platform == "{game_platform}"')['year_of_release'].min())
print('Средняя продолжительность жизни платформы: ',sum(years_of_life)// len(years_of_life),' лет')



actual_data = df.query('year_of_release >= 2010')
#Найдем наиболее прибыльные платформы за этот период
actual_platform = actual_data.pivot_table(index=['platform'], values='sum_sales', aggfunc='sum').reset_index().sort_values(by='sum_sales', ascending=False)
 
sns.set_style('whitegrid')
plt.figure(figsize=(18, 5))
sns.barplot(data = actual_platform, x='platform', y='sum_sales')
plt.show()


# Как можно заметить, за этот период больше всего продавались ps3 и xbox360, но будут ли они актуальны в последующих годах?  Сейчас проверим.

# Теперь найдем потенциально прибыльные платформы и построим график «ящик с усами» по глобальным продажам игр в разбивке по платформам.


all_platforms = actual_data['platform'].unique()
for game_platform in all_platforms:
    actual_data[actual_data['platform'] == game_platform].pivot_table(index='year_of_release', values='sum_sales', aggfunc='sum').plot(kind='bar', figsize=(15,5))
    plt.title(game_platform)
    plt.show()
    actual_data[actual_data['platform'] == game_platform].boxplot(column='sum_sales',figsize = (10,10))
    plt.show()
    display(actual_data[actual_data['platform'] == game_platform]['sum_sales'].describe())


# Такие приставки как xbox360, ps3, 3ds, wii, pc, ds, psp, psv, ps2 идут на активный спад за весь актуальный период.  
# Таким образом наиболее актуальными будут являться консоли последнего поколения ps4 и xboxOne. Сейчас по таким показателям как суммарное и среднее число продаж,из этих двух приставок, лидирует PS4. PS3 так же была более успешной в этом плане, однако максимальное число проданных копий за год принадлежит приставке xbox360, несмотря на это игры для приставки от Sony пользуются всё таки большим спросом.


samples = ['ps3','ps4','x360','xone']
for game_platform in samples:
    print('Диаграммы рассеяния для', game_platform)
    actual_data[actual_data['platform'] == game_platform].plot(x = 'critic_score', y = 'sum_sales',kind = 'scatter')
    plt.title('Диаграмма рассеяния зависимости продаж от оценок критиков')
    plt.show()
    print('Коэффициент корреляции равен: ',actual_data[actual_data['platform'] == game_platform]['critic_score'].corr(actual_data[actual_data['platform'] == game_platform]['sum_sales']))
    actual_data[actual_data['platform'] == game_platform].plot(x='user_score', y='sum_sales', kind='scatter')
    plt.title('Диаграмма рассеяния зависимости продаж от оценок пользователей')
    plt.show()
    print('Коэффициент корреляции равен: ',actual_data[actual_data['platform'] == game_platform]['user_score'].corr(actual_data[actual_data['platform'] == game_platform]['sum_sales']))



all_sales_genre = actual_data.pivot_table(index = 'genre',values = ['eu_sales','na_sales','jp_sales','other_sales','sum_sales'],aggfunc = 'sum')
display(all_sales_genre.sort_values(by = 'sum_sales', ascending=False))

# В ходе исследовательского анализа данных можно прийти в следующим выводам:  
# 1) Начиная с 2000-го года количество выпускаемых игр стабильно увеличивалось до 2009-го года. Начиная с 2010 количиство начинает либо снижаться, либо оставаться примерно на одном уровне. Больше всего игр выпускалось с 2005 по 2009 годы. Скорее всего в эти годы студии пытались узнать что больше всего нравится пользователям, поэтому выпускалось так много игр. После 2009 года у студий сложился определённый портрет потрибителя и началась работа не на количество, а на качество.  
# 2) Самыми актуальным платформами за всё время являются ps2, хbox360 и ps3.  
# 3) На основе данных по ps3, можно сказать, что пользователи больше доверяют другим пользователям, чем критикам, так как коэффициент корреляции между оценками критиков и продажами ниже, чем коэффициент корреляции  между оценками пользователей и продажами.

# Составим портрет пользователя для каждого региона. <a name="step4"></a>

# Заменим пропуски в столбце rating на rp (рейтинг ожидается).


warnings.filterwarnings("ignore")
actual_data['rating'] = actual_data['rating'].fillna('rp')




regions = ['na_sales','eu_sales','jp_sales']
categories = ['platform','genre','rating']
def top_5(data, values, category):
    print(data.pivot_table(index = category, 
                            values = values, 
                            aggfunc='sum').sort_values(by = values,
                                                         ascending = False).head())
    data.pivot_table(index = category, 
                            values = values, 
                            aggfunc='sum').sort_values(by = values,
                                                         ascending = False).head().plot(kind = 'bar')
    plt.show()
    print()
for category in categories:
    print('------'*20)
    display(f'Распределение по {category}')
    for region in regions:
        top_5(actual_data, region, category)
        #actual_data[actual_data['platform'] == game_platform].pivot_table(index='year_of_release', values='sum_sales', aggfunc='sum').plot(kind='bar', figsize=(15,5))
        #plt.title(game_platform)
        #plt.show()


# ### Вывод
# 1) NA регион.  
# В этом регионе пользователи больше всего предпочитают использовать xbox360 и ps3. И любят активные жанры игр по тип экшенов и шутеров.  
# 2) EU регион.  
# В Европе пользователи предпочитают всё те же приставки, но ps3 находится на первом месте. По жанрам картина практически идентична, за исключением того, что жанр pole-playing любят немного больше, чем misc.  
# 3) JP регион.  
# Тут Японцы полностью поддерживают своего производителя в плане приставок, топ разделили между собой приставки от Nintendo и Sony. В плане игрового жанра на первом месте расположился role-playing( в Америке и Европе он не входит в топ), а вот жанра шутер в топе вовсе нет.
#   
# Можно заметить интересную особенность в плане ограничения по рейтингу. Американских и европейских пользователей ограничение "m"(для взрослых) не останавливает, тогда, как Японцы относятся к этому более ответсвенно и на первом месте по продажам у них расположились игры с рейтингом "e"( для всех), а рейтинг "m" находится лишь на 3 месте.



warnings.filterwarnings("ignore")
actual_data.dropna(subset = ['user_score'], inplace = True)


xone_users = actual_data.query('platform =="xone"')
pc_users = actual_data.query('platform =="pc"')


# Проверим гипотезу, что средние пользовательские рейтинги платформ Xbox One и PC одинаковые.  
# Нулевая гипоетеза: средний пользовательский рейтинг Xbox One и PC равны.  
# Альтернативная: средний пользовательский рейтинг Xbox One и PC не равны.  

# Проведем тест Левене, чтобы определить насколько равны дисперсии двух выборок.



#print(levene(xone_users['user_score'], pc_users['user_score']))
print(np.var(xone_users['user_score'],ddof = 1),np.var(pc_users['user_score'],ddof = 1))



alpha = 0.05
results = st.ttest_ind(pc_users['user_score'],xone_users['user_score'],  equal_var = False)

print('p-значение: ', results.pvalue)

if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")

# Проверим гипотезу, что средние пользовательские рейтинги жанров Action и Sports равны.  
# Нулевая гипотеза: средние пользовательские рейтинги жанров Action и Sports равны.  
# Альтернативная: средние пользовательские рейтинги жанров Action и Sports не равны.


sports_game = actual_data.query('genre == "sports"')
action_game = actual_data.query('genre == "action"')
print(levene(action_game['user_score'], sports_game['user_score']))


# p-value очень мало, значит дисперсии выборок отличаются.


alpha = 0.05
results = st.ttest_ind(action_game['user_score'],sports_game['user_score'],  equal_var = False)

print('p-значение: ', results.pvalue)

if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")

# Общий вывод.
# В ходе работы были исследованы продажи интернет магазина Стримчик.  
# Как итог, можно подвести, что наибольшей популярностью пользуются игры для платформ от компаний Microsoft и Sony, меньшим спросом пользуется Nintendo.  
# Самым прибыльным является NA регион. Он лидирует по продажам практически во всех жанрах.  
# Больше всего игр выпускалось в период с 2005 по 2009 год.  
# Пользователи больше прислушиваются к мнению других пользователей, нежели к критикам.  
# Средний пользовательский рейтинг Xbox One и PC равны.
# Средние пользовательские рейтинги жанров Action и Sports равны.

