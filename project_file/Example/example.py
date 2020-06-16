from datetime import datetime
import configparser

def create_config_file(data):
    config = configparser.RawConfigParser()
    config['DEFAULT'] = data['DEFAULT']

    dt = data['data_type']
    for s in dt.keys():
        config.add_section(s)
        for k in dt[s].keys():
            config[s][k] = data['DEFAULT'][k]



    with open('conf.ini', 'w') as conf_file:
        config.write(conf_file)


#получение словаря с правильными типами данных
def filling_config_data(data, con_file):
    


conf_data = {'DEFAULT':{'tikets':'SPY', 'start_date':'01.01.2010','initial_capital':'1000.0','add_funds':'100.0'}, 'data_type':{'list':{'tikets': str.split},'date':{'start_date':datetime.strptime},'money':{'initial_capital':float, 'add_funds':float}}}
txt = 'SPY SPA'
#t = conf_data['list']['tikets'](txt)
create_config_file(conf_data)









