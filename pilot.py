from pathlib import Path
import platform
import copy
import pandas as pd
import json
import pickle
import os
import numpy as np
import csv
from sklearn.metrics import confusion_matrix

# begin path string helper functions #

# get directory part of path string
def get_dir(s):
    j = 0
    p = ''
    for i in range(len(s), 0, -1):
        if s[i-1] is '\\' or s[i-1] is '/':
            j = i
            break
    for i in range(0, j):
        p += s[i]
    return p

# returns directory part and file part of path string as tuple
def split_path(s):
    j = 0
    p = ''
    f = ''
    for i in range(len(s), 0, -1):
        if s[i-1] is '\\' or s[i-1] is '/':
            j = i
            break
    for i in range(0, j):
        p += s[i]
    for i in range(j, len(s)):
        f += s[i]
    return p, f

# gets the name of the file without extension
def get_fname(s):
    p = None
    f = None
    if '\\' in s or '/' in s:
        p, f = split_path(s)
    else:
        f = s
    fname = ''
    for i in range(0, len(f)):
        if f[i] is '.':
            break
        else:
            fname += f[i]
    return fname

# gets file type (eg bin, txt, csv, etc.)
def get_ftype(s):
    tp = ''
    p = None
    f = None
    if '\\' in s or '/' in s:
        p, f = split_path(s)
    else:
        f = s[:]
    if f is not None:
        j = 0
        for i in range(0, len(f)):
            if f[i] is '.':
                j = i
                break
        for i in range(j+1, len(f)):
            tp += f[i]
    return tp

# alternative file extension getter (eg .bin, .txt, .csv, etc.) 
def get_fext(s):
    ex = ''
    p = None
    f = None
    if '\\' in s or '/' in s:
        p, f = split_path(s)
    else:
        f = s[:]
    if f is not None:
        j = 0
        for i in range(0, len(f)):
            if f[i] is '.':
                j = i
                break
        for i in range(j, len(f)):
            ex += f[i]
    return ex

# gets user path string
def get_userpath(loading=False, prompt='Enter path to file: '):
    yn = ''
    pth = ''
    d, f = None, None
    while True:
        yn = ''
        pth = ''
        d, f = '', ''
        pth += input(prompt)
        d, f = split_path(pth)
        if not loading:
            if os.path.isfile(pth):
                o = input('File \''+f"{f}"+'\' already exists. Overwrite? (y/n): ')
                if o is 'Y' or o is 'y':
                    break
                else:
                    continue
            elif not os.path.isdir(d):
                yn += input('Invalid directory. Use local directory instead? (y/n): ')
                if yn is 'y' or yn is 'Y':
                    pth = get_local_userpath(f)
                    break
                else:
                    continue
            elif f is '' or f is None:
                print('No file specified in path')
                continue
            else:
                break
        else:
            if not os.path.isfile(pth):
                cont = input('File could not be found. Continue? (y/n): ')
                if cont is 'N' or cont is 'n':
                    return None
                else:
                    continue
            elif f is '' or f is None:
                print('No file specified in path')
                continue
            else:
                break
    return pth

# gets the local directory path of user's machine
def get_local_userpath(f, force_binary=True):
    m = ''
    if platform.system() is 'Windows':
        m += '\\'
    elif platform.system() is 'Darwin' or platform.system() is 'Linux':
        m += '/'
    else:
        raise NotImplementedError('Operating system not recognized!')
    if force_binary:
        return str(os.getcwd()) + m + get_fname(f) + '.bin'
    else:
        return str(os.getcwd()) + m + f
            
# Primary class
class NaiveBayes:
    def __init__(self, file_path=None, DFrame=None, dec_attr=None, cond_attrs=None, prior=None, c_list=None, tup=None):
        self.file_path = file_path
        self.DFrame = None
        self.dec_attr = None
        self.cond_attrs = None
        self.prior = None
        self.c_list = None
        self.tup = None
        if DFrame is None:
            self.DFrame = pd.read_csv(file_path)
            if any(pd.isna(self.DFrame[self.DFrame.columns[-1]])):
                self.DFrame = self.DFrame[self.DFrame[self.DFrame.columns[-1]].notna()]
        else:
            self.DFrame = copy.deepcopy(DFrame)
            if any(pd.isna(self.DFrame[self.DFrame.columns[-1]])):
                self.DFrame = self.DFrame[self.DFrame[self.DFrame.columns[-1]].notna()]
        if dec_attr is None:
            self.dec_attr = self.DFrame.columns[-1]
        else:
            self.dec_attr = copy.deepcopy(dec_attr)
        if cond_attrs is None:
            self.cond_attrs = self.DFrame.columns[:-1]
        else:
            self.cond_attrs = copy.deepcopy(cond_attrs)
        if prior is None:
            self.prior = self.freq(self.DFrame[self.dec_attr], 'dict')
        else:
            self.prior = copy.deepcopy(prior)
        if c_list is None:
            self.c_list = self.inverse_p(self.DFrame, self.cond_attrs, self.dec_attr)
        else:
            self.c_list = copy.deepcopy(c_list)
        if tup is None:
            self.tup = (self.prior, self.c_list, self.cond_attrs, self.dec_attr)
        else:
            self.tup = copy.deepcopy(tup)

    # converts class instance to dictionary in order to load from file more easily
    def __dict__(self):
        return { 'file_path' : self.file_path, \
            'DFrame' : self.DFrame, \
            'dec_attr' : self.dec_attr, \
            'cond_attrs' : self.cond_attrs, \
            'prior' : self.prior, \
            'c_list' : self.c_list, \
            'tup' : self.tup }

    # drops headers of a dataframe
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
                return { i: x.value_counts()[i] for i in x.unique() }
            else:
                return (x.name, { i: x.value_counts()[i] for i in x.unique()})
        return pd.DataFrame([x.value_counts()[i] for i in x.unique()], index=x.unique(), columns=[x.name])

    def cond_p(self, df, c, d):
        C = df.groupby(c).groups
        D = df.groupby(d).groups
        # The + 1.0 attempts to midigate what is called the frequency problem of naive Bayes classification,
        # where if the frequency of a condition occurring is zero, then the conditional probability will also
        # be zero, and so the algorithm will not be able to properly classify that particular attribute given the test data
        freq_problem = False
        for j in D.keys():
            for i in C.keys():
                if int((C[i] & D[j]).size) <= 0:
                    freq_problem = True
                    break
        P_DC = None
        if freq_problem:
            P_DC = { (i, j): (float((C[i] & D[j]).size) + 1.0)/ float(C[i].size)  for i in C.keys() for j in D.keys() }
        else:
            P_DC = { (i, j): (C[i] & D[j]).size / C[i].size for i in C.keys() for j in D.keys() }
        return P_DC  #returns P(d|c) as a dict

    def inverse_p(self, df, cond_list, decision_list):
        """ Build a list of dict of inverse probabilities
        """
        p_list = [self.cond_p(df, decision_list, i) for i in cond_list] #build a list of dicts
        return p_list

    # save model utility function
    def SaveModel(self, s, as_bin=True):
        forbidden_chars = '<>\":\'|?*'
        valid_path = False
        valid_file = False
        
        multi_slash = False
        multi_period_inpath = False
        multi_period_infile = False
        forbidden_char_inpath = False
        forbidden_char_infile = False
        
        p, filename = split_path(s)
        
        # checks for valid path string depending on the operating system
        if platform.system() is 'Windows':
            if '\\\\' in p:
                valid_path = False
                multi_slash = True
            if lambda forbidden_chars, p: any(i in forbidden_chars for i in p):
                valid_path = False
                forbidden_char_inpath = True
            if '..\\' in p:
                if '..\\' is not p[0:2]:
                    valid_path - False
                    multi_period_inpath = True
            else:
                valid_path = True
            if filename.count('.') > 1:
                valid_file = False
                multi_period_infile = True
            if lambda forbidden_chars, filename: any(i in forbidden_chars for i in filename if i is not ':'):
                valid_file = False
                forbidden_char_infile = True
            if p.count(':') > 1:
                valid_file = False
                forbidden_char_infile = True
            else:
                valid_file = True
        elif platform.system() is 'Linux' or platform.system() is 'Darwin':
            if '//' in p:
                valid_path = False
                multi_slash = True
            if '../' in p:
                if '../' is not p[0:2]:
                    valid_path = False
                    multi_period_inpath = True
            else:
                valid_path = True
            if filename.count('.') > 1:
                valid_file = False
                multi_period_infile = True
            else:
                valid_file = True
        else:
            raise NotImplementedError('Operating system not recognized!')
        if valid_path is False or valid_file is False:
            if valid_path is False:
                print('Invalid path name:')
                if multi_slash:
                    print('-extra \'\\\' or \'/\' in path')
                if multi_period_inpath:
                    print('-improper relative path syntax')
                if forbidden_char_inpath:
                    print('-use of system-reserved characters in path')
            if valid_file is False:
                print('Invalid file name:')
                if multi_period_infile or forbidden_char_infile:
                    print('-use of system-reserved characters in file')
            return False
        else:
            # if valid path string, save the file depending the extension
            if os.path.isdir(p):
                try:
                    if get_ftype(filename) is 'bin':
                        with open(s, 'wb') as f:
                            pickle.dump(self.__dict__(), f)
                    elif get_ftype(filename) is 'txt' and as_bin:
                        with open(s, 'wb') as f:
                            f.write(str(self.__dict__()))
                    elif get_ftype(filename) is 'txt' and not as_bin:
                        with open(s, 'w') as f:
                            f.write(str(self.__dict__()))
                    elif get_ftype(filename) is 'json' and as_bin:
                        with open(s, 'wb') as f:
                            json.dump(self.__dict__(), f)
                    elif get_ftype(filename) is 'json' and not as_bin:
                        with open(s, 'w') as f:
                            f.write(json.dumps(self.__dict__()))
                    else:
                        with open(s, 'wb') as f:
                            pickle.dump(self.__dict__(), f)
                except (pickle.PicklingError, json.JSONEncodeError) as e:
                    print('Model failed to save to file: ', e)
                    return False
                else:
                    print('Model saved successfully.')
                    return True

    # load the model using a class method so no instances of a model need to exist
    @classmethod
    def LoadModel(cls, s):
        bin_str_ex = False
        txt_bin_ex = False
        bin_bin_ex = False
        txt_str_ex = False
        ldbin = None
        pth, filename = split_path(s)
        if not os.path.exists(s):
            print('file not found')
            return False
        else:
            if get_ftype is 'bin':
                with open(s, 'rb') as f:
                    ldbin = pickle.load(f)
            elif get_ftype is 'json':
                try:
                    with open(s, 'r') as f:
                        ldbin = json.load(f)
                except json.JSONDecodeError as dec_err:
                    txt_bin_ex = True
                    try:
                        with open(s, 'r') as f:
                            ldbin = json.loads(f.read())
                    except json.JSONDecodeError as dec_err:
                        txt_str_ex = True
                try:
                    with open(s, 'rb') as f:
                        ldbin = json.load(f)
                except json.JSONDecodeError as dec_err:
                    bin_bin_ex = True
                    try:
                        with open(s, 'rb') as f:
                            ldbin = json.loads(f.read())
                    except json.JSONDecodeError as dec_err:
                        bin_str_ex = True
                finally:
                    if all([bin_str_ex, txt_str_ex, bin_bin_ex, txt_bin_ex]):
                        print('could not load object from json file')
                        return None
            else:
                with open(s, 'rb') as f:
                    ldbin = pickle.load(f)
        if ldbin is not None:
            data = copy.deepcopy(ldbin)
            print('\nModel loaded successfully.\n')
            return cls(**data)
        else:
            return None

    def Print(self):
        print(self.DFrame)

    def GetNumberOfRows(self, df=None):
        if df is None:
            return len(self.DFrame.index)
        else:
            return len(df.index)

    # was causing problems so this function is no longer used
    def ValidateTestList(self, L):
        return all([L[i] in self.DFrame[self.DFrame.columns[i]].unique() for i in range(0,len(L))])
    
    # The classification function
    def Classify(self, L, N):
        best_cat = None
        if len(L) != len(self.DFrame.columns) - 1:
            print('Shape Error: input file data does not match shape of model\'s training data')
            return None
        # if not self.ValidateTestList(L):
            # print('Validation Error: input file data does not match format of model\'s training data')
            # return None
        else:
            c_probs = []
            numElements = N
            best = 0.0
            # for each unique decision,
            for dec in self.prior.keys():
                probs=[]
                # decision and frequency tuple in list
                t = [(dec, i) for i in L]
                for d in self.c_list:
                    for k, v in d.items():
                        # for every particular combination of attributes that lead to that decision for a list of features,
                        # append that to the list of inverse conditional probabilities 
                        if k in t:
                            probs.append(v)
                # compute probability of this particular list of features being in the class as denoted as the unique decision
                r = 1.0
                for p in probs:
                    r *= p
                c_prob = float(r)*float(self.prior[dec])/float(numElements)
                c_probs.append(c_prob)
                # this blocks finds the best-fitting class for the list of features
                if c_prob > best:
                    best = c_prob
                    best_cat = dec
        return best_cat

    # different methods of implementing the Classify function
    def TestClassifierList(self, df):
        if len(df[0]) != len(df.columns) - 1:
            print('input file data does not match shape of model\'s training data')
            return None
        return [self.Classify(row, len(df)) for row in df]
    def TestClassifier(self, df):
        if len(df.columns) != len(self.DFrame.columns):
            print('input file data does not match shape of model\'s training data')
            return None
        truth_values = df[df.columns[-1]].tolist()
        df = df.drop(df.columns[-1], axis = 1)
        test_data = df.values.tolist()
        return truth_values, [self.Classify(row, len(df.index)) for row in test_data]
    def TestClassifierWithHeaders(self, df):
        if len(df.columns) != len(self.DFrame.columns):
            print('input file data does not match shape of model\'s training data')
            return None
        truth_values = df[df.columns[-1]].tolist()
        # df = df.drop(df.index[0], axis = 0)
        df = df.drop(df.columns[-1], axis = 1)
        test_data = df.values.tolist()
        return truth_values, [self.Classify(row, len(df.index)) for row in test_data]



# Primary driver program
class Program:
    # Nested custom confusion matrix wrapper class so it prints in a more easily-read format
    class ConfusionTable:
        ylabel = 'Actual'
        xlabel = 'Predicted'
        def __init__(self, class_labels=None, Matrix=None, Table=None):
            self.ClassLabels = None
            non_str = False
            for i in class_labels:
                if not isinstance(i, str):
                    non_str = True
                    break
            if non_str:
                self.ClassLabels = [str(i) for i in class_labels]
            else:
                self.ClassLabels = class_labels[:]

            self.Matrix = [[i for i in row] for row in Matrix]
            self.Header = [j for i in [[self.ylabel], self.ClassLabels] for j in i]
            self.Table = Table
            if self.Table is None:
                self.Table = pd.DataFrame([[j] + [i for i in row] for j , row in zip(self.ClassLabels, self.Matrix)], columns=self.Header[:])
        def Print(self):
            print(''.join(' ' for i in range(0, len(''.join(self.ylabel)))) + ''.join(' ' for i in range(0, len(''.join(self.Header))//2))+''.join(self.xlabel))
            print(self.Table)

    def __init__(self):
        self.state = None
        self.model = None
        self.TestData = None
        self.TestResults = None
        self.ConfusionMatrix = None
        self.quit = False
    
    # Main driver loop
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

    # Different thunk functions for each menu choice
    def Learn(self):
        print('  1. Learn a Na'+'\u00ef'+'ve Bayesian Classifier from categorical data')
        self.model = NaiveBayes(self.GetUserFilePathString())
        print('model training complete')
    def Save(self):
        print('  2. Save a model')
        if self.model == None:
            print('There are no models currently loaded. Please load a model or create one')
            return
        self.model.SaveModel(get_userpath())
    def Load(self):
        print('  3. Load a model and test its accuracy')
        if self.model != None:
            del self.model
            self.model = None
        self.model = NaiveBayes.LoadModel(get_userpath(loading=True, prompt='Enter the path to load the model: '))
        s = get_userpath(loading=True, prompt='Enter the path to the testing data: ')
        yn = input('Does the table contain headers? (y/n): ')
        if yn == 'y' or yn == 'Y':
            self.TestData = self.LoadCSVFile(s)
            if any(pd.isna(self.TestData[self.TestData.columns[-1]])):
                self.TestData = self.TestData[self.TestData[self.TestData.columns[-1]].notna()]
            self.TestResults = self.model.TestClassifierWithHeaders(self.TestData)
        else:
            self.TestData = self.LoadCSVFile(s, header=self.model.DFrame.columns.to_list())
            if any(pd.isna(self.TestData[self.TestData.columns[-1]])):
                self.TestData = self.TestData[self.TestData[self.TestData.columns[-1]].notna()]
            self.TestResults = self.model.TestClassifier(self.TestData)
        if self.TestResults is None:
            print('Error in performing test on model')
            return
        else:
            # Construct confusion matrix
            actuals, predictions = self.TestResults
            self.ConfusionMatrix = self.ConfusionTable(self.model.DFrame[self.model.DFrame.columns[-1]].unique().tolist(), confusion_matrix(actuals, predictions, labels=self.model.DFrame[self.model.DFrame.columns[-1]].unique()))
            print(self.ConfusionMatrix.Header)
            print('\nConfusion Matrix:\n')
            self.ConfusionMatrix.Print()
            print()
    def Interactive(self):
        if self.model == None:
            print('There are no models currently loaded. Please load a model or create one')
            yn = input('Would you like to load a model? (y/n): ')
            if yn is 'Y' or yn is 'y':
                self.model = NaiveBayes.LoadModel(get_userpath(loading=True, prompt='Enter the path to load the model: '))
            else:
                return
        while True:
            print('    1. Enter a new case interactively')
            s = input('Please enter a test case for the model separated by spaces or \'q\' to quit:\n')
            if s == 'q':
                print('    2. Quit')
                break
            non_str = False
            for i in s:
                if not isinstance(i, str):
                    non_str = True
                    break
            testList = None
            if non_str:
                testList = [str(i) for i in s]
            else:
                testList = s.split()
            print('Classification: ' + str(self.model.Classify(testList, len(testList))))

    # Menu printing utilities
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

    # User input utilities
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
    
    # Various other helper functions
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

    def LoadCSVFile(self, f, header=None):
        if Path(f).exists() and self.IsCSV(self.FileFormatStringFromPath(f)):
            if header is not None:
                return pd.read_csv(f, names=header)
            else:
                return pd.read_csv(f)

    def LoadTestData(self, s):
        if os.path.exists(s) and self.IsCSV(self.FileFormatStringFromPath(s)):
            df = None
            with open(s, 'r') as f:
                reader = csv.reader(f)
                df = list(reader)
            return df






# Driver instantiation and main loop invocation
if __name__ == "__main__":
    program = Program()
    program.Run()