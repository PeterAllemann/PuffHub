import xml.etree.ElementTree as ET
import numpy as np

gxlFilePath = "data/gxl/"


class Molecule:

  def __init__(self, id,assigned):
    gxlFile = gxlFilePath + str(id) + ".gxl"
    self.tree = ET.parse(gxlFile)
    self.id = id

    self.atoms =  atoms = list()

    for node in self.tree.findall(".//string"): #path to and Atom see https://docs.python.org/2/library/xml.etree.elementtree.html
        node_value = node.text
        atoms.append(node_value.strip())   #there might be useless white space

    self.atomCount = len(self.atoms)
    self.adj = createAdj(self.tree, self.atomCount) #saving all needed edgesd72
    self.assigned = setLabel(assigned);


def setLabel(label):
  if label == 'i':
    return 0;
  if label == 'a':
    return 1;

  return -1;


  #consider leaving the middle row out, not shure if an atom could have such paths
def createAdj(tree,atomCount):
  adj = np.zeros((atomCount,atomCount));

  for e in tree.findall(".//edge"):

    start = int(e.get('from')[1]);
    end = int(e.get('to')[1]);
    adj[start - 1, end - 1] = 1;
    adj[end - 1, start - 1] = 1;

  print(adj)
  return adj


mol = Molecule(35,0)


print("molecule added")