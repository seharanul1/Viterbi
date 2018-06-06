


import math
import numpy as np
import hmm
#import read_input as ii


#############################
###bigram 파일 읽기
#############################
f1 = open("bigram.txt", 'r')
ss=0
bigram = {}


while True:
    read = f1.readline()
    if not read : break
    read = read[:-1]
    line = read.split('\t')
    if line[0] == '' or line[0] == ' ' or line[0] == '\n': break
    bigram[(line[0],line[1])] = float(line[2])
    ss+=float(line[2])

    
###############################
###input 파일 읽기
###############################
"""read = f1.readline()
read = f1.readline()[1:-2]
print(read[:-3])
print(read.endswith('lab'))
f1.close()
f2 = open(read[:-3]+"txt",'r')
mfcc=f2.read().split()[2:]
f2.close()"""

f1 = open("reference.txt", 'r')
data =[]
#i=0
while True:
    read = f1.readline()
    if not read : break
    read = read[1:-2]
    if read.endswith('lab'):
        #i=i+1
        ad = read[:-3]+"txt"
        f = open(ad,'r')
        mfcc=f.read().split()[2:]
        data.append((ad,mfcc))
        f.close()
f1.close()



phone_state = {}
phone_trans = {}
PHONES = hmm.phones
for phone in PHONES:
    phone_state[phone[0]]=phone[2]
    phone_trans[phone[0]]=phone[1]


"""
for phone in PHONES:
    table = phone_trans[phone[0]]
    #print([table[i][len(table[0])-1] for i in range(len(table))])
    #print("==========")"""



"""state1 = phone_state['f'][0]
state2 = phone_state['f'][1]
state3 = phone_state['f'][2]
pdf1 = state1[0][0]
pdf2=state1[0][1]
pdf3=state1[0][2]
weight, mean, var = pdf1[0], pdf1[1], pdf1[2] 
print(weight)"""



class word:
    def __init__(self,phone,start):
        self.n_phones = len(phone)
        self.phones = phone
        self.start_prob = start
        self.a = {} #phone끼리 건너가는 transition prob
        self.trans_table = [phone_trans[p] for p in self.phones] #각 phone의 a들
        for idx in range(self.n_phones-1):
            table = self.trans_table[idx]
            next_table = self.trans_table[idx+1]
            end_prob = table[len(table)-2][len(table[0])-1]
            str_prob = next_table[0][1]
            self.a[(self.phones[idx],self.phones[idx+1])]=end_prob*str_prob
        self.last_phone=self.trans_table[self.n_phones-1]
        self.end_prob = self.last_phone[len(self.last_phone)-2][len(self.last_phone[0])-1]  
        if self.phones[-1] == 'sp':
            self.end = ('sp', 2)
        else:
            self.end = ('sil', 4)


word_dict = {
    '<s>':word(['sil'],0.990000),
    'eight':word(['ey', 't', 'sp'],0.000925),
'five':word(['f', 'ay', 'v', 'sp'],0.000890),
'four':word(['f', 'ao','r', 'sp'],0.000886),
'nine':word(['n', 'ay', 'n' ,'sp'],0.000905),
'oh':word(['ow', 'sp'],0.000968),
'one':word (['w', 'ah', 'n', 'sp'],0.000905) ,
'seven':word (['s', 'eh', 'v', 'ah', 'n', 'sp' ],0.000869) ,
'six':word (['s', 'ih', 'k', 's' ,'sp'],0.000939),
'three':word(['th', 'r', 'iy', 'sp'],0.000883),
'two':word(['t', 'uw', 'sp'],0.000941),
'zero':word(['z', 'ih', 'r', 'ow', 'sp'],0.000889),
'zero_':word(['z', 'iy', 'r', 'ow', 'sp'],0.000889)
}



word_list = ['<s>', 'eight','five','four','nine','oh','one' ,'seven' ,'six' ,'three' ,'two' ,'zero' ,'zero_' ]

###################
###HMM
###################
HMM=[]
#HMM=[('start',0,0)]
for word in word_list:
    for idx, phone in enumerate(word_dict[word].phones):
        if phone == 'sp':
            N = 3
        else:
            N=5
        for n in range(N):
            HMM.append((word,idx,n))
#HMM.append(('end',0,0))
lg0 = math.inf*(-1)



def phone_end(x):
    if x == 'sp':
        return 2
    else:
        return 4


###################
###tansition matrix a
###################
# ('<s>', 'sil', 0)   ('<s>', 'sil', 1)
# start ('start',0,0) end ('end',0,0)
def get_a(sfrom, sto):
    (fw, fph_idx, fs) = sfrom
    (tw, tph_idx, ts) = sto
    if fw == 'start' and (0,0) == (tph_idx, ts):
        return math.log(word_dict[tw].start_prob)
    elif fw == 'start':
        return lg0
    fph = word_dict[fw].phones[fph_idx]
    if tw == 'end' and word_dict[fw].end == (fph,fs):
        return math.log(1)
    elif tw == 'end':
        return lg0
    tph = word_dict[tw].phones[tph_idx]
    if fw == tw and fph_idx == tph_idx:
        out = phone_trans[fph][fs][ts]
        #return math.log(phone_trans[fph][fs][ts])
    elif fw == tw and fph_idx+1==tph_idx and (fph,tph) in word_dict[fw].a and fs == phone_end(fph) and ts == 0:
        out = word_dict[fw].a[(fph,tph)]
        #return math.log(word_dict[fw].a[(fph,tph)])
    elif fw != tw and word_dict[fw].end == (fph,fs) and (0,0) == (tph_idx, ts):
        if fw=='zero_' :
            fw = 'zero'
        if tw == 'zero_':
            tw = 'zero'
        if (fw,tw) not in bigram:
            out = 0
        else :
            out = bigram[(fw,tw)]**0.1
        #return math.log(ii.bigram[(fw,tw)])
    else:
        #print("a000000 : ",sfrom, "=>",sto)
        out = 0
    if out == 0:
        return lg0
    return(math.log(out))



def find(state, vit):
    for v in vit:
        if v[0] == state:
            return v
for phone in PHONES:
    states = phone_state[phone[0]] #phone='f'
    for sn in range(len(states)):
        state = states[sn]

        pdfs = [state[0][i] for i in range(10)]

        l = []
        for idx, pdf in enumerate(pdfs):
            weight, mean, var = pdf[0], pdf[1], pdf[2]
            ##########################
            pro_var = 1
            for v in var:
                pro_var = pro_var * v
            #print(pro_var, math.log(pro_var))

            sum1 = math.log(weight)-(39/2)*math.log(2*math.pi)-math.log(pro_var)
            if (len(pdf) < 4):
                phone_state[phone[0]][sn][0][idx].append(sum1)
            else:
                phone_state[phone[0]][sn][0][idx] = [weight, mean, var, sum1]

###################
####Gaussian mixture b
###################
def get_b(phone, sn, x):
    states = phone_state[phone] #phone='f'
    sn = sn-1
    #0,1,2,3,4 
    #-1,0,1,2,3 
    if len(states) == 3 and (sn<0 or sn==3):
        return 0
    elif len(states) == 1 and (sn != 0):
        return 0
    state = states[sn]
    
    pdfs = [state[0][i] for i in range(10)]
    
    l = []
    for pdf in pdfs:
        weight, mean, var, sum1 = pdf[0], pdf[1], pdf[2], pdf[3]
        ##########################
        temp3 = [((float(x[i])-mean[i])/(var[i]))**2 for i in range(len(x))]
        sum2 = np.sum(temp3)*(-0.5)
        
        
        #print(temp3,"\n===",temp2)
        rslt = sum1 + sum2
        #print(rslt)
        ##########################
        #print(rslt, np.long(rslt))
        l.append(rslt)
    #print(l)
    tt=0
    for i in range(1,len(pdfs)):
        #print(l[i]-l[0])
        try:
            #print(l[i],l[0],(l[i]-l[0]))
            temp = math.exp((l[i] - l[0]))
        except OverflowError:
            temp = 1
        tt = tt+temp
        
    b=l[0]+math.log(tt+1) 
    return b*0.1


############
##Viterbi
############
START=('start',0,0)
END=('end',0,0) 
def viterbi(x) :     
    T=len(x)
    #delta[t] = [[(state,p,psi),],]
    vtb = [[] for _ in range(T+1)]
    #psi[t] = [(w,ph,s1):w]
    #psi =[None for _ in range(T)]
 
    for state in HMM:
        prob = get_a(('start',0,0),state) + get_b(word_dict[state[0]].phones[state[1]] ,state[2],x[0]) 
        if get_a(('start',0,0),state) != lg0  :
            #if get_a(('start',0,0),state)!= 1:
            vtb[0].append((state,prob,START))
            #psi[0][state]=state[0]
    print(0,":",len(vtb[0]))
    """print(vtb[0])  
    print(len(vtb[0]))
    print("===") """
    for t in range(1, T):
        for state in HMM:
            max_p=(None, None)
            for d in vtb[t-1]:
                prob = d[1] + get_a(d[0],state) + get_b(word_dict[state[0]].phones[state[1]] ,state[2],x[t])                    
                if get_a(d[0],state) != lg0 and (max_p[1] == None or max_p[0] < prob):
                    max_p = (prob, d[0])
                    #print("prev:",d[0], state, get_a(d[0],state), get_b(word_dict[state[0]].phones[state[1]] ,state[2],x[t]),prob   )                  
            if max_p[1] != None:
                vtb[t].append((state, max_p[0], max_p[1]))
                #print((state, max_p[0], max_p[1]))
        #print(t,len(vtb[t]))
        #print(vtb[t])
        if t % 50 == 0:
            print(t,":",len(vtb[t]))
        #print(vtb[t],"\n===============\n")
        """if len(vtb[t]) > 224:
            print(t,":",vtb[t],"\=============================n\n")"""
    """print(T)            
    print(len(vtb[1]))
    print(vtb[1]) """
    #print(vtb[T-1])
    state = END
    max_p=(vtb[T-1][0][1], vtb[T-1][0][0])
    for d in vtb[T-1]:
        if max_p[0] < d[1]:
            max_p = (d[1], d[0])
    vtb[T].append((state, max_p[0], max_p[1]))    
    #print(vtb[T])
    #print(vtb[T][0][2])

    back_vtb_state = END
    ##################
    ph_seq=[]
    ##################
    for t in range(T,-1,-1):
        cur_vtb = find(back_vtb_state,vtb[t])
        ph_seq.append(cur_vtb[0])
        back_vtb_state=cur_vtb[2]
        #print(cur_vtb)
    ph_seq.reverse()
    #[('<s>', 0, 0), ('<s>', 0, 1), ('<s>', 0, 1), ('end', 0, 0)]
    #fw, fph, fs ,...   fs == phone_end(fph) and ts == 0
    word_seq = []
    for i, state in enumerate(ph_seq):
        if state[0] == 'end':
            break
        next_state=ph_seq[i+1]
        (fw, fph_idx, fsn) = state
        (tw,tph_idx,tsn) = next_state
        if fw != tw or (fph_idx !=0 and tph_idx==0):
            word_seq.append(fw)
    word_seq = [word for word in word_seq if word != '<s>']
    for i,_ in enumerate(word_seq):
        if word_seq[i] == 'zero_':
            word_seq[i] = 'zero'
        
    return ph_seq, word_seq

    """print(delta[0])
    print(len(delta[0]))
    #print(psi[0])"""


###########################################################
###전체 data 중 sample_size 수 만큼의 파일만 viterbi로 계산
###속도가 너무 느려 현재는 1개의 파일에 대해서만 계산
###########################################################
sample_size=1


f2 = open("recognized.txt", 'w')
f2.write("#!MLF!#")
for idx,d in enumerate(data):
    ad = '\n"'+ d[0][:-3] +'rec"'
    f2.write(ad)
    
    if idx < sample_size:
        dim = 39
        cut_data = []
        for i in range(0,len(d[1]),dim):
            cut_data.append(d[1][i:i+dim])
        seq, out = viterbi(cut_data)
        for word in out:
            f2.write("\n"+word)
            print(""+word)
    f2.write("\n.")
    print(".")
    print(seq)
f2.close()


"""for state in ph_seq:
    (word, ph_idx, sn) = state
    phone = word_dict[word].phones[ph_idx]
    end_phone = len(word_dict[word].phones)-1
    if ph_idx == end_phone and sn == phone_end(phone):
        word_seq.append(word)"""



"""for phone in PHONES:
    print(phone[0], "=\n")
    for p in phone_trans[phone[0]]:
        print(p)"""



"""for phone in PHONES:
    states = phone_state[phone[0]]
    print(len(states))"""



"""for phone in PHONES:
    states = phone_state[phone[0]] #phone='f'
    for sn in range(len(states)):
        sn = sn-1
        #0,1,2,3,4 
        #-1,0,1,2,3 
        if len(states) == 3 and (sn<0 or sn==3):
            continue
        elif len(states) == 1 and (sn != 0):
            continue
        state = states[sn]

        pdfs = [state[0][i] for i in range(10)]

        l = []
        for idx, pdf in enumerate(pdfs):
            print(phone[0],pdf[3])"""




