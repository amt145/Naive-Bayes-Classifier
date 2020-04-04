from pathlib import Path
import pandas as pd
import json
import pickle
import uuid
import os
import numpy as np
import csv
from sklearn.metrics import confusion_matrix


""" 
def SaveModel():
    usr_in = None
    while True:
        usr_in = input('Please enter file name in csv-format: ')
        if (' ' in usr_in) or ('<' in usr_in) or ('>' in usr_in) or (':' in usr_in) or ('\"' in usr_in) or ('/' in usr_in) or ('\\' in usr_in) \
        or ('|' in usr_in) or ('?' in usr_in) or ('*' in usr_in) or ('&' in usr_in):
            print('\nFile name cannot contain reserved characters')
            continue
        else:
            break
"""        
class NaiveBayes:
    def __init__(self, file_path=None):
        # self.guid = uuid.uuid4()
        self.file_path = file_path
        self.DFrame = pd.read_csv(file_path)
        if any(pd.isna(self.DFrame[self.DFrame.columns[-1]])):
            self.DFrame = self.DFrame[self.DFrame[self.DFrame.columns[-1]].notna()]
        self.dec_attr = self.DFrame.columns[-1]
        self.cond_attrs = self.DFrame.columns[:-1]
        self.prior = self.freq(self.DFrame[self.dec_attr], 'dict')
        print(type(self.prior))
        for i in self.prior:
            print(type(i), type(self.prior[i]))
        self.c_list = self.inverse_prob(self.DFrame, self.cond_attrs, self.dec_attr)
        print(type(self.c_list))
        self.tup = (self.prior, self.c_list, self.cond_attrs, self.dec_attr)
        #self.TestData = None

    class UUIDEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, uuid.UUID):
                # if the obj is uuid, we simply return the value of uuid
                #return obj.hex
                return obj.int
            return json.JSONEncoder.default(self, obj)

    def to_dict(self):
        #return { 'guid': self.guid, 'file_path': self.file_path, 'DFrame': None, 'dec_attr': self.dec_attr, 'cond_attrs': self.cond_attrs, 'prior': self.prior, 'c_list': self.c_list, 'tup': self.tup, 'TestData': self.TestData }
        pr, c, cond, dec = self.tup
        return { 'file_path': self.file_path, 'DFrame': self.DFrame.to_json(), 'dec_attr': self.dec_attr, 'cond_attrs': self.cond_attrs.tolist(), 'prior': self.prior, 'c_list': self.c_list, 'tup': (pr, c, cond.tolist(), dec) }
    def DropHeaders(self, df):
        df.pop(0)
        return df

    def freq(self, x, opt='DataFrame'):
        """ x is a Series
            it returns a DataFrame (by default) indexed by unique values of x and
            their frequency counts
        """
        if opt != 'DataFrame':
            if opt == 'dict':
                return { i: int(x.value_counts()[i]) for i in x.unique()}
            else:
                return (x.name, { i: x.value_counts()[i] for i in x.unique()})
        return pd.DataFrame([x.value_counts()[i] for i in x.unique()], index=x.unique(), columns=[x.name])

    def attr_freqdict(self, attr_series):
        return { i: attr_series.value_counts()[i] for i in attr_series.unique() }

    def cond_prob(self, frame, cond_attrs, dec_attr):
        cond_groups = frame.groupby(cond_attrs).groups
        dec_group = frame.groupby(dec_attr).groups
        return { (i, j): (cond_groups[i] & dec_group[j]).size / cond_groups[i].size for i in cond_groups.keys() for j in dec_group.keys() }

    def inverse_prob(self, frame, cond_attrs, dec_attr):
        return [self.cond_prob(frame, dec_attr, i) for i in cond_attrs]

    def SaveModel(self, s):
        """
        #if not Path(s).exists():
        #    print('path does not exist')
        #    return None
        f_str = ''
        j = 0
        pathdir=os.path.dirname(os.path.abspath(__file__))
        f = None
        for i in range(0, len(s)):
            if s[i] == '.':
                j = i
                break
        if '.' not in s:
            f_str += 'bin'
        else:
            for i in range(j, len(s)):
                f_str += s[i]
        if f_str == 'bin':
            if '.' not in s:
                f = open(os.path.join(pathdir, s+'.bin'), 'wb')
            else:
                f = open(os.path.join(pathdir, s), 'wb')
            #f.write(str(self.__dict__).encode('utf-8'))
            #f.write(json.dumps(self.to_dict().encode('utf-8'), cls=self.UUIDEncoder))
            f.write(json.dumps(self.to_dict()).encode('utf-8'))
            f.close()
        else:
            f = open(os.path.join(pathdir, s), 'w')
            #f.write(str(self.__dict__))
            #f.write(json.dumps(self.to_dict(), cls=self.UUIDEncoder))
            #print(json.dumps(self.to_dict()))
            f.write(json.dumps(self.to_dict()))
            f.close()
        """

        # for now, unconditionally save object as a binary

        # list comprehension does not support 'break'
        fs = ''
        for i in s:
            if i is '.':
                break
            else:
                fs += i
        fs += '.bin'
        


    @classmethod
    def LoadModel(cls, s):
        #if not Path(s).exists():
        #    return None
        f_str = ''
        j = 0
        pathdir=os.path.dirname(os.path.abspath(__file__))
        f = None
        jdict = None
        for i in range(0, len(s)):
            if s[i] == '.':
                j = i
                break
        for i in range(j, len(s)):
            f_str += s[i]
        print(f_str)
        print(os.path.join(pathdir, s))
        if 'bin' in f_str:
            f = open(os.path.join(pathdir, s), 'rb')
            jdict = json.loads(f.read().decode('utf-8').replace('\'', '\"'))
            f.close()
        elif 'json' in f_str:
            f = open(os.path.join(pathdir, s), 'r')
            jdict = json.loads(f.read().replace('\'', '\"'))
            f.close()
        if jdict is not None:
            return cls(**jdict)
        else:
            print('object could not be created from file')
            return None

    def Print(self):
        print(self.DFrame)

    def GetNumberOfRows(self):
        return len(self.DFrame.index)

    def ValidateTestList(self, L):
        return all([L[i] in self.DFrame[self.DFrame.columns[i]].unique() for i in range(0,len(L))])
    
    def Classify(self, L):
        best_cat = None
        if len(L) != len(self.DFrame.columns) - 1:
            print('input file data does not match shape of model\'s training data')
            return None
        if not self.ValidateTestList(L):
            print('input file data does not match format of model\'s training data')
            return None
        else:
            c_probs = []
            numElements = self.GetNumberOfRows()
            best = 0.0
            for dec in self.prior.keys():
                probs=[]
                t = [(dec, i) for i in L]
                for d in self.c_list:
                    for k, v in d.items():
                        if k in t:
                            probs.append(v)
                r = 1.0
                for p in probs:
                    r *= p
                c_prob = float(r)*float(self.prior[dec])/float(numElements)
                c_probs.append(c_prob)
                if c_prob > best:
                    best = c_prob
                    best_cat = dec
        return best_cat

    def TestClassifierList(self, df):
        if len(df[0]) != len(df.columns) - 1:
            print('input file data does not match shape of model\'s training data')
            return None
        return [self.Classify(row) for row in df]
    def TestClassifier(self, df):
        if len(df.columns) != len(self.DFrame.columns):
            print('input file data does not match shape of model\'s training data')
            return None
        truth_values = df[df.columns[-1]].tolist()
        df = df.drop(df.columns[-1], axis = 1)
        test_data = df.values.tolist()
        return truth_values, [self.Classify(row) for row in test_data]

    def TestClassifierWithHeaders(self, df):
        if len(df.columns) != len(self.DFrame.columns):
            print('input file data does not match shape of model\'s training data')
            return None
        truth_values = df[df.columns[-1]].tolist()
        df = df.drop(df.index[0], axis = 0)
        df = df.drop(df.columns[-1], axis = 1)
        test_data = df.values.tolist()
        return truth_values, [self.Classify(row) for row in test_data]




class Program:
    def __init__(self):
        self.state = None
        self.model = None
        self.TestData = None
        self.TestResults = None
        self.ConfusionMatrix = None
        self.quit = False

    def Run(self):
        while True:
            self.PrintMenu()
            c = self.GetMenuChoice()
            if c == 1:
                self.Learn()
            elif c == 2:
                self.Save()
            elif c == 3:
                self.Load()
            elif c == 4:
                self.PrintSubmenu()
                c2 = self.GetSubmenuChoice()
                if c2 == 1:
                    self.Interactive()
                else:
                    break
            elif c == 5:
                break

    def Learn(self):
        print('  1. Learn a Na'+'\u00ef'+'ve Bayesian Classifier from categorical data')
        self.model = NaiveBayes(self.GetUserFilePathString())
        print('model training complete')

    def Save(self):
        print('  2. Save a model')
        if self.model == None:
            print('There are no models currently loaded. Please load a model or create one')
            return
        while True:
            s = input('please enter the file name to save the model: ')
            if Path(s).exists():
                self.model.SaveModel(s)
                break
            elif not Path(s).exists():
                #use relative path
                pathdir=os.path.dirname(os.path.abspath(__file__))
                fname = os.path.join(pathdir, s)
                print(fname)
                self.model.SaveModel(fname)
                break
            else:
                print('directory does not exist')
    
    def Load(self):
        print('  3. Load a model and test its accuracy')
        if self.model != None:
            del self.model
            self.model = None
        while True:
            s = input('please enter the file name to load the model: ')
            if Path(s).exists():
                self.model = NaiveBayes.LoadModel(s)
                break
            elif not Path(s).exists():
                #use relative path
                pathdir=os.path.dirname(os.path.abspath(__file__))
                fname = os.path.join(pathdir, s)
                print(fname)
                self.model = NaiveBayes.LoadModel(fname)
                break
            else:
                print('file does not exist')
        yn = input('does the table contain headers? (y/n): ')
        self.TestData = self.LoadCSVFile(s)
        if yn == 'y' or yn == 'Y':
            self.TestResults = self.model.TestClassifierWithHeaders(s)
        else:
            self.TestResults = self.model.TestClassifier(s)
        actuals, predictions = self.TestResults
        self.ConfusionMatrix = confusion_matrix(actuals, predictions, labels=[self.model.dec_attr.unique()])
    
    def Interactive(self):
        if self.model == None:
            print('There are no models currently loaded. Please load a model or create one')
            return
        while True:
            print('    1. Enter a new case interactively')
            s = input('please enter a test case for the model separated by spaces or \'q\' to quit')
            if s == 'q':
                print('    2. Quit')
                break
            print(self.model.Classify(s.split()))

    def PrintMenu(self):
        print('\nMain Menu')
        print('1. Learn a Na'+'\u00ef'+'ve Bayesian Classifier from categorical data')
        print('2. Save a model')
        print('3. Load a model and test its accuracy')
        print('4. Apply a Na'+'\u00ef'+'ve Bayesian Classifier to new cases interactively')
        print('5. Quit')
        
    def PrintSubmenu(self):
        print('  4. Apply a Na'+'\u00ef'+'ve Bayesian Classifier to new cases interactively')
        print('    1. Enter a new case interactively')
        print('    2. Quit')

    def GetMenuChoice(self):
        usr_in = None
        while True:
            usr_in = input()
            if len(usr_in) == 1:
                if int(usr_in) == 1 or int(usr_in) == 2 or int(usr_in) == 3 or int(usr_in) == 4 or int(usr_in) == 5:
                    break
        return int(usr_in)

    def GetSubmenuChoice(self):
        usr_in = None
        while True:
            usr_in = input()
            if len(usr_in) == 1:
                if int(usr_in) == 1 or int(usr_in) == 2:
                    break
        return int(usr_in)

    def GetUserMenuChoice(self):
        self.PrintMenu()
        usr_in = self.GetMenuChoice()
        usr_in2 = None
        if usr_in == 4:
            self.PrintSubmenu()
            usr_in2 = self.GetSubmenuChoice()
            return usr_in2
        else:
            return usr_in
        
    def GetUserFilePathString(self):
        usr_in = None
        while True:
            usr_in = input('Please enter file name in csv-format: ')
            usr_format = ''
            usr_file_str = ''
            f_index = 0
            last_s_index = 0
            for i in range(0,len(usr_in)):
                if usr_in[i] == '/' or usr_in[i] == '\\':
                    last_s_index = i
                if usr_in[i] == '.':
                    f_index = i+1
                    break
            for i in range(f_index, len(usr_in)):
                usr_format += usr_in[i]
            for i in range(last_s_index, f_index-1):
                usr_file_str += usr_in[i]
            usr_file_str += '.' + usr_format
            if not Path(usr_in).exists():
                print('File \"'+usr_file_str+'\" does not exist.')
                cont = input('Continue? (y/n): ')
                if cont == 'n' or cont == 'N':
                    usr_in = ''
                    break
            else:
                break
        return  usr_in
    
    def FileStringFromPath(self, s):
        file_str = ''
        last_s_index = 0
        for i in range(0, len(s)):
            if s[i] == '/' or s[i] == '\\':
                last_s_index = i
        for i in range(last_s_index+1, len(s)):
            file_str += s[i]
        return file_str
    def FileFormatStringFromPath(self, s):
        format_str = ''
        for i in range(s.find('.')+1, len(s)):
            format_str += s[i]
        return format_str

    def IsCSV(self, s):
        if s == 'csv':
            return True
        else:
            return False

    def LoadCSV(self):
        usr_in = self.GetUserFilePathString()
        if usr_in != '' and self.IsCSV(self.FileFormatStringFromPath(usr_in)):
            return pd.read_csv(usr_in)
    def LoadCSVFile(self, f):
        if Path(f).exists() and self.IsCSV(self.FileFormatStringFromPath(f)):
            return pd.read_csv(f)

    def LoadTestData(self, s):
        if Path(s).exists and self.IsCSV(self.FileFormatStringFromPath(s)):
            df = None
            with open(s, 'r') as f:
                reader = csv.reader(f)
                df = list(reader)
            return df







program = Program()
program.Run()

# nb1 = NaiveBayes('Project2/TestData/weather.csv')
# print(nb1.prior)
# print(nb1.c_list)
# tests = [['rainy', 'hot', 'high', False], ['rainy', 'hot', 'normal', False], ['sunny', 'cool', 'normal', True]]
# truth_values = ['yes', 'no', 'yes']
# predictions = nb1.TestClassifier(tests)
# print(predictions)
# cm = confusion_matrix(truth_values, predictions)
# print(cm)