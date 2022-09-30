import zipfile
import re
import datetime

"""

@author: Alex Andrzejek

"""

#edit this if we need different format
dateformat = '{2:04}-{0:02}-{1:02}' # 0 month, 1 day, 2 year
date_time = True 

old_archive = zipfile.ZipFile('csv.zip', 'r')
new_archive = zipfile.ZipFile('csv_new.zip', 'w', zipfile.ZIP_DEFLATED)


for filename in old_archive.namelist():
    print(filename)
    file = old_archive.open(filename) 
    filetext = ''   
    
    for line in file:
        #print(line)
        newline = ''
        date_result = re.match('^([\s \S]+\,)(\d{1,2}\/\d{1,2}\/\d{4})\s(\d{1,2}\:\d{1,2}\:\d+)(\,[\s \S]+)', line.decode('utf-8')) 
        
        
        if date_result:
            dt = datetime.datetime.strptime(date_result.group(2), '%m/%d/%Y')
            if date_time:    
                newline = date_result.group(1)+dateformat.format(dt.month, dt.day, dt.year)+' '+date_result.group(3)+date_result.group(4)
            else:
                newline = date_result.group(1)+dateformat.format(dt.month, dt.day, dt.year)+','+date_result.group(3)+date_result.group(4)
                
        else:
            newline = line.decode('utf-8')

        
        path_result = re.match('^([\s \S]+\,\")([\S \s]+\")(\,[\s \S]+)', newline);
        
        if path_result:
            newline = path_result.group(1)+'.'+path_result.group(2)+path_result.group(3)
        else:
            newline = newline
        
        filetext += newline
        
    new_archive.writestr(filename, filetext)            
        
old_archive.close()
new_archive.close()     
        


#dt = datetime.datetime.strptime("3/21/2014", '%m/%d/%Y')
#print('{2:04}-{1}-{0:02}'.format(dt.month, dt.day, dt.year))