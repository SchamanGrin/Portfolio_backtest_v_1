﻿Проект создан для того, что бы разобраться как делать системы работы с биржевыми данными.
Тема 1. Тестирование стратегий (Backtasting)
Создан набор классов.
Data - работа с данными
event - работа с событиями
execution - обработка взаимодействия событий порождаемых портфелем и ответных событий рынка
ib_execution - класс для взаимодействия с api interactive brocker
portfolio - определение и методы работы портфеля. Создание ордеров в ответ на события и сигналы стратегии
strategy - определение и методы по созданию сигналов на покупку / продажу в соответствии со стратегией
perfomancer - определения и методы оценки результативности портфеля

Текущие задачи:

1.  (Done) Сейчас дата передается в формате год-месяц-день. По хорошему нужно передавать год-месяцдень час:минута:секунда
Добавлять строки стоит при перед конвертацией в индекс
2. (Done) Написать обработчик комиссий для IB. Комиссии за неактивность и комиссии за сделку. Временное решение - добавленеи расчета коммисий в модуль event
3. (Done)Написать обработку ввода даты, которой нет в датасете. Временно решение - исключение, если дата не в датасете.
        Обработчик внесен в модуль класс data
4. Написать стратегию покупки раз в две недели, пока не кончатся деньги
    4.1 Done. Написать процедуру пополнения два раза  в классе "портфель"
    4.2 Написать процедуру подачи сигнала на покупку два раза в месяц. 1го и 15го, или ближайшим к ним торговым датам

5. Написать методику оценки эффективности портфеля с учетом регулярного пополнения (XIRR, коэффициент Шарпа, максимальную просадку, TWRR)
    5.1 (done) Загрузить тестовые данные из эксель, для проверки расчета функции XIRR
    5.2 написать функцию XIRR на с использованием scipy
    5.2.1 Расчитать полную доходность, исходя из сумм поступлений / суммы расходов
    5.2.2 Расчитать доходность по годам
    5.2.3 добавить преобразование из таблицы с casflow в используемый список кортежей
    5.3 Написать формирование таблицы денежного потока для расчета доходности. Должна быть отдельная таблица, учитывающая денежные поступления и итоговую стоимость
    5.3.1 (done) выделить отдельный dataframe для cashflow с индексами по датам и заполнить его первую строку.
    5.4 Написать расчет оценки риска для портфлея с нерегулярными пополнениями (Статья на https://rostsber.ru/publish/stocks/python_asset.html)
    5.5  (done)Написать расчет взвешанной по времени нормы доходности TWRR, расчитать доходность портфеля на каждый день. (Статья на https://fin-accounting.ru/cfa/l1/quantitative/cfa-portfolio-return-measurement)
    5.5.1 (done) протестировать корректность расчета TWRR
    5.6 (done) Сделать единый интерфейс, в который передавать в DataFrame даты, стоимость портеля/бумаги на эту дату и денежный поток на дату с соответствующим знаком.
        Выводить twrr и xirr одним значением и для twrr выводить доходность на каждую дату, для xirr выводить на каждую дату доходность, считая
        что каждая дата - день получения дохода на полную стоимость порфеля, не прибавляя к нем денежный поток на эту дату
    5.7 В едином интерфейсе расчета доходности, сделать возможность выбирать период данных




6. (done) Вынести модуль подготовки файла данных в отдельный модуль - модуль alpha

7. (done) Сделать чистые репозиторий и нстроить работу с ним Git

8. Сделать конфигурационный файл с ключами,который не должен попадать на git
8.1 (done) разобратсья с тем, как задавать относительный путь к файлу с ключем (https://python-scripts.com/pathlib)
8.2 (отклонено) Оформить хранение зашифрованного ключа (https://python-scripts.com/encryption-cryptography). Ключи нет смысла хранить в зашифрованном виде, лучше просто не тянуть ключ
на гит. а для нового пользователя, создавать новый конфиг и просить его заполнить.
8.3  (done) написать создание конфига для ключей, если его нет


9. Создать конфигурационный файл для хранение входных параметров (https://habr.com/ru/post/485236/ , https://zhevak.wordpress.com/2015/01/08/%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0-%D1%81-%D0%BA%D0%BE%D0%BD%D1%84%D0%B8%D0%B3%D0%B0%D0%BC%D0%B8-%D0%B2-%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B0%D1%85-%D0%BD%D0%B0-python/)
9.1 (done) организовать хранение стартовой даты и стартового капитала
9.3 (done)сделать шаблон конфигурационного файла, для его создания
9.3 (done) Преобразовать конфиг входных параметров, что бы по сектору преобразовывать тип данных




10. (done) Сделать библиотеку, формирующую конфигурационные файлы

11.1 Сделать окна для запроса ключа или файла с ключем (https://younglinux.info/tkinter/dialogbox.php, https://pythonspot.com/tk-file-dialogs/)
11.2

12. (done)Перейди с datetime на numpy datetime (https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.datetime.html, https://www.geeksforgeeks.org/python-numpy-datetime64-method/#:~:text=With%20the%20help%20of%20numpy,datetime64()%20method.&text=Return%20%3A%20Return%20the%20date%20in,yyyy%2Dmm%2Ddd' )

13. Написать логику расчета приказа на покупку
13.1




