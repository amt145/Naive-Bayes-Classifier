import json
import os
class Person:
    def __init__(self, first, last, number, address):
        self.first = first
        self.last = last
        self.number = number
        self.address = address

    def to_dict(self):
        return self.__dict__


    def save_json(self):
        """
        f = open(f'/Projects/Project2/{self.first}.json', 'w')
        f.write(json.dumps(self.__dict__))
        f.close()
        """
        import os
        # THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{self.first}.json')
        f = open(my_file, 'w')
        f.write(json.dumps(self.__dict__))
        f.close()


    @classmethod
    def from_json(cls, json_string):
        jdict = json.loads(json_string)
        return cls(**jdict)


    @classmethod
    def LoadModel(cls, s):
        """
        f_str = ''
        j = 0
        for i in range(0, len(s)):
            if s[i] == '.':
                j = i
                break
        for i in range(j, len(s)):
            f_str += s[i]
        if f_str == 'bin':
        """
        f = open(s, 'r')
        raw_data = f.read()
        f.close()
        jdict = json.loads(json.loads(raw_data.decode('utf-8'), encoding='utf-8'))
        return cls(**jdict)

person = Person('Alex', 'T', 1, 111)
pathstr = os.path.dirname(os.path.abspath(__file__))
print(pathstr)

f = open(os.path.join(pathstr,'me.bin'), 'wb')
f.write(str(person.__dict__).encode('utf-8'))
f.close()

f_b = open(os.path.join(pathstr,'me.bin'), 'rb')
d = json.loads(f_b.read().decode('utf-8').replace('\'', '\"'))
f_b.close()
print(d)

person2 = Person('Alex', 'T', 1, 111)
f2 = open(os.path.join(pathstr,'me.txt'), 'w')
f2.write(str(person.__dict__))
f2.close()

f22 = open(os.path.join(pathstr,'me.txt'), 'r')
p22 = json.loads(f22.read().replace('\'', '\"'))
f22.close()
# print(p22)