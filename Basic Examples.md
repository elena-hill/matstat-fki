# Введение в Python

Это блокнот - удобное средство для проведения экспериментов и анализа данных.
Так можно оформлять текст, создавать заголовки.


```python
# так можно устанавливать библиотеки
pip install seaborn
```


      File "<ipython-input-1-a79c0a35f353>", line 2
        pip install seaborn
                  ^
    SyntaxError: invalid syntax
    


## Стандартные классы в Python


```python
int_var = 5 # целое число
float_var = 0.53 # вещественное число
str_var = "string" # строка
bool_var = True # логический тип
list_var = [0, 1, 2] # список
tuple_var = (0, 1, 2) # кортеж
set_var = {0, 1, 2} # множество 
dict_var = {'a': 0, 'b': 1, 'c': 2} # словарь
```

Важно, что в Python существуют изменяемые и неизменяемые переменные. К неизменяемым относятся int, str, bool, tuple. Иными словами, когда вы присваиваете их значение другим переменным, создается новый объект. Изменить вы неизменяемые переменные не можете.


```python
str_var2 = str_var
str_var += " another" # ~ str_var = str_var + " another"
str_var2
```




    'string'



Изменяемые же объекты ведут себя иначе. Их можно изменять. При присваивании их значение другим переменным объекты не копируются, а копируются только ссылки на объект. 


```python
# например

list_var2 = list_var
list_var2 += [3] # однако лучше использовать list_var2.append(3)
list_var
```




    [0, 1, 2, 3]



## Вывод данных

Выводить данные можно самыми раными методами, остановимся на 2х стандартных. Результат выполнения последней операции выводится на экран как в примере выше (если вам этого не хочется, поставьте в самом конце ';'), а вывести переменную из середины кода в ячейке можно с помощью функции print.


```python
print("42 is a magic number")
```

    42 is a magic number
    


```python
"42 is a magic number"
```




    '42 is a magic number'



Но последний метод не будет работать, если в конце будет не просто объект, а объект внутри условия или цикла.


```python
for i in range(1):
    "42 is a magic number"
```

## Функции

Здесь функции не требуют объявления типов входных и выходных значений в силу динамической типизации.


```python
def product(a, b):
    return a * b
```

Важно!!! Отступы являются частью синтаксиса (возможно, это даже приятнее чем видеть бесконечные {}).


```python
product(7, 6)
```




    42




```python
product('7', 6) # да, тут можно умножить строку на число :)
```




    '777777'



## Циклы и генераторы

Есть 2 типа циклов: прохождение по итерируемому объекту (с помощью for) или совершение действий пока выполнено какое-то условие (цикл while).


```python
lst = []
for i in range(5): # это специальный итерируемый объект, выдающий числа 0, 1, 2, 3, 4
    lst.append(i) # добавляем элемент в список
lst
```




    [0, 1, 2, 3, 4]




```python
# иной способ генерации такого списка
lst = [i for i in range(5)]
lst
```




    [0, 1, 2, 3, 4]




```python
lst = []
i = 0
while i < 5:
    lst.append(i)
    i += 1
lst
```




    [0, 1, 2, 3, 4]




```python
# к итерируемым объектам относятся список, кортеж, словарь, множество, строка

for c in "Колмогоров":
    print(c)
```

    К
    о
    л
    м
    о
    г
    о
    р
    о
    в
    

Можно увидеть, что именно отступы определяют то, где начинается тело цикла и где оно заканчивается.
При этом отступы должны состоять либо из знаков табуляции, либо из пробелов.

## Условия


```python
a = 1
if a == 1:
    b = 2
else:
    b = 3
b
```




    2




```python
# можно писать условия в одну строчку
b = 2 if a == 1 else 3
b
```




    2



Разберемся в том, что тут происходит.
"a == 1" - условие
Если оно верно, то возвращается 2,
иначе возвращается 3.
То есть общая конструкция такова:


```python
"условие истинно" if 5 < 6 else "условие ложно"
```




    'условие истинно'




```python
"условие истинно" if 5 > 6 else "условие ложно"
```




    'условие ложно'




```python
# множественное условие
a = 1
b = 2

if a < 1:
    c = 4
elif b > 2.5:
    c = 3
else:
    c = 2
c
```




    2



## Ошибки

Мы часто ошибаемся, поэтому нужно знать, как исправляться или как игнорировать ошибки).


```python
a = 3
if a > 2:
print("a is more then 2")
# TypeError - класс ошибки, позволяет понять о сути самой проблемы
```


      File "<ipython-input-20-60c5ce86bae6>", line 3
        print("a is more then 2")
            ^
    IndentationError: expected an indented block
    



```python
try:
    a = 3
    if a > 2:
    print("a is more then 2")
except Exception as er: # отлавливаем ошибку
    print(er)
```


      File "<ipython-input-21-03490d863ae1>", line 4
        print("a is more then 2")
            ^
    IndentationError: expected an indented block
    


Увы, такие ошибки вы просто должны не совершать)


```python
a = "3" + 4
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-22-c83c0461db39> in <module>
    ----> 1 a = "3" + 4
    

    TypeError: must be str, not int



```python
try:
    a = "3" + 4
except Exception as er: # отлавливаем ошибку
    print(er)
```

    must be str, not int
    

# Векторы и матрицы. Библиотека numpy

## Инициализация векторов

Обычно для работы с векторами используют библиотеку numpy. Она позволяет делать многие базовые операции из линейной алгебры, и не только.


```python
# импортируем библиотеку
import numpy as np
```


```python
# создаем вектор из списка
a = np.array([1, 2, 3])
a
```




    array([1, 2, 3])




```python
# генерируем с помощью имеющихся библиотечных средств
b = np.arange(3)
b
```




    array([0, 1, 2])




```python
# обращение к элементам вектора
a[0]
```




    1




```python
b[:-1] #не включая последний
```




    array([0, 1])



## Арифметические операции


```python
a + b
```




    array([1, 3, 5])




```python
# скалярное произведение векторов

a.dot(b)
```




    8




```python
# поэлементные дейтсвия над элементами вектора

(2 * b)**2
```




    array([ 0,  4, 16])




```python
# агрегирующие функции

a.sum()
```




    6




```python
b.mean()
```




    1.0



## Инициализация матриц


```python
# создаем матрицу из списка списков
A = np.array([[4, 0, 3], 
              [3, 5, -1]])
A
```




    array([[ 4,  0,  3],
           [ 3,  5, -1]])




```python
B = np.vstack([a, b])
B
```




    array([[1, 2, 3],
           [0, 1, 2]])



## Операции из линейной алгебры


```python
# поэлементное умножение
A * B
```




    array([[ 4,  0,  9],
           [ 0,  5, -2]])




```python
#транспонирование
B.T
```




    array([[1, 0],
           [2, 1],
           [3, 2]])




```python
# матричное умножение
A.dot(B.T)
```




    array([[13,  6],
           [10,  3]])




```python
A + B
```




    array([[5, 2, 6],
           [3, 6, 1]])




```python
# обратная матрица
C = np.array([[0, 1],
              [-1, 0]])
np.linalg.inv(C)
```




    array([[-0., -1.],
           [ 1.,  0.]])



# Таблицы. Библиотека pandas

Классическая библиотека для работы с таблицами. Позволяет делать многие естественные операции с таблицами (наподобие SQL'ных).

## Инициализация таблиц


```python
import pandas as pd
```


```python
# из словаря
df = pd.DataFrame(data={"Score": [1, 2], "Name": ["Petr", "Ivan"]})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Petr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivan</td>
    </tr>
  </tbody>
</table>
</div>




```python
# из списка списков
df2 = pd.DataFrame([[1, "Petr"], [2, "Ivan"]], columns=["Score", "Name"])
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Petr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivan</td>
    </tr>
  </tbody>
</table>
</div>



Строки и стоблцы таблицы являются объектами класса Series.


```python
df["Score"]
```




    0    1
    1    2
    Name: Score, dtype: int64




```python
# так можно посмотреть на тип объекта
type(df["Score"])
```




    pandas.core.series.Series



## Получение данных из таблицы


```python
# Выделение строк по условию
df.loc[df["Score"] > 1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivan</td>
    </tr>
  </tbody>
</table>
</div>




```python
# аггрегирующие функции
df["Score"].mean()
```




    1.5




```python
df["Score"].max()
```




    2



## Методы apply, assign

Достаточно часто приходится сталкиваться с тем, что нам нужно из какого-то стоблца сделать другой, или же использовать информацию из всей таблицы сразу. Для этого можно использовать циклы, но это крайне нежелательно, так как есть методы, позволяющие ускорить такого рода действия.


```python
# лямбда-функции позволяют создавать безымянные функции прямо внутри методов
df["Score"].apply(lambda x: x**2)
```




    0    1
    1    4
    Name: Score, dtype: int64




```python
df.assign(cnt=1, is_Petr=lambda _df: _df["Name"] == "Petr")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Name</th>
      <th>cnt</th>
      <th>is_Petr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Petr</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivan</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Группировка и агрегирующие функции

В большинстве случаев мы имеем большой набор данных, который нам нужно сгруппировать по определенному признаку, а затем получить по нему статистику некоторыми агрегирующими функциями.


```python
df = pd.DataFrame([[True, "M", False],
                   [False, "W", False],
                   [True, "M", True],
                   [True, "W", False]],
                  columns=["Went to store", "Sex", "Bought something"])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Went to store</th>
      <th>Sex</th>
      <th>Bought something</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>M</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>W</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>M</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>W</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# с помощью слэшей можно переносить код со строчки на строчку
# 1 оставляем только те строки, где в столбце стоит True
# 2 добавляем новый столбец тождественную единицу для удобства подсчёта
# 3 группируем по полу
# 4 суммируем значения некоторых столбцов в каждой группе
# 5 добавляем новый столбец, означающий эмперическую вероятность покупки для каждой группы
# 6 выбираем определенный столбец
df.loc[df["Went to store"]] \
  .assign(cnt=1) \
  .groupby("Sex") \
  .agg({"Bought something": "sum", "cnt": "sum"}) \
  .assign(conversion=lambda _df: _df["Bought something"] / _df["cnt"]) \
  ["conversion"]
```




    Sex
    M    0.5
    W    0.0
    Name: conversion, dtype: float64



# Графики. Библиотека matplotlib


```python
# импортируем модуль библиотеки
import matplotlib.pyplot as plt
```


```python
# обычный график
a = np.linspace(0, 1)
plt.plot(a, a**2, '^')
```




    [<matplotlib.lines.Line2D at 0x7fe030dfb748>]




![png](Basic%20Examples_files/Basic%20Examples_88_1.png)



```python
# гистограмма
data = pd.DataFrame(data={"Name": ["Maria", "Petr", "Maria", "Ivan", "Ivan"],
                          "Point": [5, 2, 3, 4, 5]})
avg_points = data.groupby("Name") \
                 .agg({"Point": "mean"})
plt.bar(avg_points.index, avg_points["Point"])
```




    <BarContainer object of 3 artists>




![png](Basic%20Examples_files/Basic%20Examples_89_1.png)



```python
# круговая диаграмма
data.assign(cnt=1) \
    .groupby("Point") \
    .agg({"cnt": "sum"}) \
    .plot(kind="pie", subplots=True) 
# можно получать графики из pandas не прибегая явно к matplotlib
# это быстрее и проще, но не всегда достаточно
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x7fe030d4f4a8>],
          dtype=object)




![png](Basic%20Examples_files/Basic%20Examples_90_1.png)


# Визуализация данных. Библиотека seaborn


```python
# импортируем модуль библиотеки
import seaborn as sb
```

В библиотеке seaborn есть набор массивов данных, которыми удобно пользоваться для обучения.


```python
iris = sb.load_dataset("iris")
iris
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>146</td>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>149</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>



Построим гистограмму для sepal_length с помощью distplot().


```python
set1 = iris['sepal_length']
sb.distplot(set1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x3b7d622288>




![png](Basic%20Examples_files/Basic%20Examples_96_1.png)


Можно построить несколько гистограмм на одном графике.


```python
set2 = iris['sepal_width']
for dataset in [set1,set2]:
    sb.distplot(dataset)
```


![png](Basic%20Examples_files/Basic%20Examples_98_0.png)


## Диаграмма рассеяния

Для построения диаграмм рассеяния будем использовать seaborn.scatterplot
https://seaborn.pydata.org/generated/seaborn.scatterplot.html


```python
sb.scatterplot(x='sepal_length', y='sepal_width', data=iris)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x3b041b1a88>




![png](Basic%20Examples_files/Basic%20Examples_101_1.png)



```python
sb.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x3b02f74548>




![png](Basic%20Examples_files/Basic%20Examples_102_1.png)



```python
sb.scatterplot(x='sepal_length', y='sepal_width', hue = 'petal_length', style='species', data=iris)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x3b0410c2c8>




![png](Basic%20Examples_files/Basic%20Examples_103_1.png)


## Ядерная оценка плотности
https://seaborn.pydata.org/generated/seaborn.kdeplot.html?highlight=kdeplot#seaborn.kdeplot


```python
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
sb.kdeplot(x, y);
```


![png](Basic%20Examples_files/Basic%20Examples_105_0.png)



```python
sb.kdeplot(x, y, n_levels=30, cmap="Purples_d");
```


![png](Basic%20Examples_files/Basic%20Examples_106_0.png)



```python
sb.kdeplot(x, bw=.15);
```


![png](Basic%20Examples_files/Basic%20Examples_107_0.png)


## Диаграмма размаха


```python
sb.boxplot(x="species", y="sepal_length", data=iris)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x3b042200c8>




![png](Basic%20Examples_files/Basic%20Examples_109_1.png)


## Параллельные координаты


```python
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(6, 5)) #зададим размеры картинки (попробуйте поменять числа)
parallel_coordinates(iris, 'species', colormap='gist_rainbow')  
```




    <matplotlib.axes._subplots.AxesSubplot at 0x3b026d13c8>




![png](Basic%20Examples_files/Basic%20Examples_111_1.png)


colormape отвечает за цвета, бывает еще, например,'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper'


```python

```
