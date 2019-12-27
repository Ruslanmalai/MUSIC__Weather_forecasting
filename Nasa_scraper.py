import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
import time
from datetime import datetime
import pandas as pd

class Nasa_scraper(object):
    
    def __init__(self, start, end):
        self.start_day = '_{}m{}{}_'.format(start[0:4], start[4:6], start[-2:])
        self.end_day = '_{}m{}{}_'.format(end[0:4], end[4:6], end[-2:])
        self.url = 'https://ozonewatch.gsfc.nasa.gov/data/omps/Y2019/'
        self.useful_files = self.get_files()
        self.long_idx = 0
        self.lat_idx = 2 +  15*133 + 4
        self.ozone, self.date_time = self.get_data()

        
    def get_files(self):
        # Makes a list of names of nasa files with appropriate dates
        html = urlopen(self.url)
        bs = BeautifulSoup(html.read(), 'html.parser')
        file_names = bs.find_all('a', {'href':re.compile('\.txt')})
        names = [name['href'] for name in file_names]
        i = 0
        for doc in names:
            if self.start_day in doc:
                idx_start = i
            elif self.end_day in doc:
                idx_end = i
                break
            i += 1
            
        files = names[idx_start:idx_end+1]
        return files 
    
    def get_data(self):
        # Gets ozone layer data for appropriate coordinates
        ozone = []
        date_time = []
        #start = time.time()
        for file in self.useful_files:
            html = urlopen('https://ozonewatch.gsfc.nasa.gov/data/omps/Y2019/' + file) # Open particular HTML
            txt = html.read().decode('utf-8').split('\n')       # read html, convert unto string and split \n
            date_line = txt[0].split()
            date = "".join(date_line[2:5])
            datetime_object = datetime.strptime(date, '%b%d,%Y')
            date_time.append(datetime_object)
            line = txt[self.lat_idx].split()
            number = int(line[0][0:3])
            ozone.append(number)
            
        #end = time.time()
        #print(end-start)
        return ozone, date_time
    
    def give_ozone_df(self):
        df = pd.DataFrame(data = self.ozone, index = self.date_time, columns = ['nasa_ozone'])
        return df
    