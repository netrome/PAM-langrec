import glob
import sys
import os
import re
import csv

# Get globals

# define paths
place = os.getcwd()
meta_path = place + "/meta/"
pattern_path = place + "/sounds/"
data_path = place + "/small_raw/"
meta = [["Language", "Age", "Sex"]] # Going to become a csv file

# Clear directories
os.system("rm sounds/*")
os.system("rm meta/*")

# Make pattern, matches any of the data files and extracts recording name and nationality
pattern = "([^\s\.]*)\.(\w\w)[aA]"
file_info_re = re.compile(pattern)
sex_re = re.compile("SEX:\s*(\w*)")
age_re = re.compile("AGE:\s*(\d*)")

#String for sox command
cmd = "sox -t raw -c 1 -e a-law -r 8000 {inFile} -e signed-integer -b 16 {out}.wav"

k = 0
n = 10000
for path, dirs, files in os.walk(data_path):
    if len(files) > 0 and k < n:
        for i in files:
                res = file_info_re.match(i)
                if res:
                    # Extract meta data
                    if res.group(2).isupper():
                        meta_file = res.group(1) + "." + res.group(2) + "O"
                    else:
                        meta_file = res.group(1) + "." + res.group(2) + "o"
                    meta_file = open(path + "/" + meta_file, "r", encoding="latin-1")
                    meta_text = meta_file.read()
                    #print(meta_text)
                    sex = sex_re.search(meta_text).group(1)
                    age = age_re.search(meta_text).group(1)
                    
                    # If we have a match, then convert file
                    file_path = path+"/"+i
                    out_path = pattern_path + str(k)
                    os.system(cmd.format(inFile = file_path, out = out_path))
                    k += 1
                    meta.append([file_path[-3:-1].upper(), age, sex])
    elif k == n:
        break

writer = csv.writer(open("meta/meta.csv", "w"))
writer.writerows(meta)

