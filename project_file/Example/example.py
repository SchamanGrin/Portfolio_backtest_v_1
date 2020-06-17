from datetime import datetime
import configparser

def to_date(x):
    return datetime.strptime(x, '%d.%m.%Y')


def create_config_file(data, path):
    config = configparser.ConfigParser()
    dt = data['data_type']
    for s in dt.keys():
        config.add_section(s)
        for k in dt[s].keys():
            config[s][k] = data['DEFAULT'][k]

    with open(path, 'w') as conf_file:
        config.write(conf_file)



# получение словаря с правильными типами данных
def filling_config_data(data, path):

    config = configparser.ConfigParser()
    config.read(path)
    dt = data['data_type']
    conf_data = {s: {o: dt[s][o](v) for o, v in config.items(s)} for s in config.sections()}

    """for s in config.sections():
        print(f'[{s}]')
        for o in config[s].keys():
            print(f'{o} = {config[s][o]}')"""

    return conf_data

conf_data = {
    'DEFAULT': {'tickets': 'SPY', 'start_date': '01.01.2010', 'initial_capital': '1000.0', 'add_funds': '100.0'},
    'data_type': {'list': {'tickets': str.split}, 'date': {'start_date': to_date},
                  'money': {'initial_capital': float, 'add_funds': float}}
}

p = 'conf.ini'
k = 'key.ini'


create_config_file(conf_data, p)
s = filling_config_data(conf_data,p)
print(s)
