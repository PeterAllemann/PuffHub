import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linear_sum_assignment as lsm
import time
import os

gxlFilePath = "data/gxl/"

#we use moleculs several times so we store them in a class with all the needed attribute
#id is  the number from the gxl file and from the valid or train set, assigned i if inactive , a if active , defautl value jsut for testing
class Molecule:

  def __init__(self, id,assigned = "null"):
    gxlFile = gxlFilePath + str(id) + ".gxl"
    tree = ET.parse(gxlFile)
    self.id = id

    self.atoms = list()

    for node in tree.findall(".//string"): #path to and Atom see https://docs.python.org/2/library/xml.etree.elementtree.html
        node_value = node.text
        #there might be useless white space
        self.atoms.append(''.join(str(ord(c)) for c in node_value.strip()))


    self.atoms = np.array(self.atoms).astype(float) #numbers are better than strings
    self.atomCount = len(self.atoms)
    self.label = setLabel(assigned);
    self.degrees = getDegrees(tree, self.atomCount) #saving all needed edges



def setLabel(label):
  label  = label.strip()
  if label == 'i':
    return 0;
  if label == 'a':
    return 1;

  return -1;



def getDegrees(tree,atomCount):
  adj = np.zeros((atomCount,atomCount));

  for e in tree.findall(".//edge"):
    start = int(e.get('from')[1]) - 1
    end = int(e.get('to')[1]) - 1
    adj[start, end] = 1
    adj[end, start] = 1

  return np.sum(adj, axis=0)


def loadMoleculesFromFile(filename):
    mols = list()
    file = open('./data/' + filename + '.txt' , "r")
    for line in file:
      #print(str.split(line, " "))
      id, label = str.split(line, " ")
      mols.append(Molecule(id, label))

    return mols


def loadMoleculesFromFileDullPath(filename):
  mols = list()
  file = open('./data/' + filename + '.txt', "r")
  for line in file:
    # print(str.split(line, " "))
    id, label = str.split(line, " ")
    mols.append(Molecule(id, label))

  return mols




def __init__(self, id,assigned = "null"):
    gxlFile = gxlFilePath + str(id) + ".gxl"
    tree = ET.parse(gxlFile)
    self.id = id

    self.atoms = list()
def ged(mol1, mol2, ce, cn):

  n = mol1.atomCount
  m = mol2.atomCount
  l = n + m

  deg1 = mol1.degrees
  deg2 = mol2.degrees



  costMat = np.zeros((l, l))

  # up right part
  mat = (np.outer (np.ones(m),(mol1.atoms))).T - mol2.atoms
  mat[mat !=  0] = 2*cn

  mat += np.abs((np.outer(np.ones(m),(deg1))).T - deg2)
  costMat[:n, :m] = mat

  # uR upper right part
  uR = np.diag(cn + ce * deg1)
  uR[uR == 0] = 4096
  costMat[:n, m:] = uR

  # lL lower left part
  lL = np.diag(cn + ce * deg2)
  lL[lL == 0] = 4096
  costMat[n:, :m] = lL

  #no need to care about the other

  #see https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
  #rowMatching are just the number 1 to n
  rowMatching, colMatching = lsm(costMat)


  ged = np.sum(costMat[rowMatching, colMatching])
  return ged


#get the lowest labels and retrn the most common of them
def prediction(k, dist, train):

    indexes = np.argpartition(dist, k)[0:k]


    nearest = np.array([train[i].label for i in indexes])

    predicted = np.array(np.argmax(np.bincount(nearest)))

    return predicted


class MoleculeVal:

  def __init__(self,path , name):


    self.id = name.replace('.gxl','')
    gxlFile = path + name
    tree = ET.parse(gxlFile)
    self.atoms = list()

    for node in tree.findall(".//string"): #path to and Atom see https://docs.python.org/2/library/xml.etree.elementtree.html
        node_value = node.text
        #there might be useless white space
        self.atoms.append(''.join(str(ord(c)) for c in node_value.strip()))


    self.atoms = np.array(self.atoms).astype(float) #numbers are better than strings
    self.atomCount = len(self.atoms)
    self.degrees = getDegrees(tree, self.atomCount) #saving all needed edges


#load the molecules
valid = loadMoleculesFromFile('valid')
train = loadMoleculesFromFile('train')


train = valid + train

Validation = list()
path = "gxl/"
for filename in os.listdir(path):
    if filename.endswith(".gxl"):

      Validation.append(MoleculeVal(path , filename))

print(Validation)





# cost arrays and ks
Cn = [1]
Ce = [1]
k = 3 # added 2 and 4 after good results with 3

#save results
name = "PuffGirlsOutPut"



for n in Cn:
  for e in  Ce:
    dist = np.zeros((len(Validation),len(train)))
    print('calc dists for cn=' + str(n) + ' ce=' + str(e))
    start = time.time()


    for j in range(0 ,len(Validation)-1):
      for i in range(0, len(train) - 1):
        dist[j,i] = ged(Validation[j], train[i], n, e)

      predicted = prediction(k, dist[j], train)
      print(predicted)

      label = ""

      if(predicted == 0):
        label = 'i'
      else:
        label = 'a'

      file = open(name, "a")
      file.write("\n" + Validation[j].id + "," + label)
      file.close()








