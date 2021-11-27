#from Mechanics.fenics_.src.genrve import *
from gmshModel.Model import RandomInclusionRVE
import numpy as np
import os 
from copy import deepcopy

def gen(seed=2, size=0.1, L=4, r=0.5, Vf=40):
    np.random.seed(seed)
    area = L*L
    inclusion_area  = area * Vf / 100
    print(inclusion_area)
    num_inclusion = int(inclusion_area / (np.pi*r**2))
    print(num_inclusion)
    #size  = [L, L, 0]
    #sets = [0.5, 2]
    initParameters={                                                                
        "inclusionSets": [r, num_inclusion],
        "inclusionType": "Circle",                                                  
        "size": [4, 4, 0],                                                          
        "origin": [0, 0, 0],                                                        
        "periodicityFlags": [1, 1, 1],                                              
        "domainGroup": "domain",                                                    
        "inclusionGroup": "inclusions",                                             
        "gmshConfigChanges": {"General.Terminal": 0,                                
                              "Mesh.CharacteristicLengthExtendFromBoundary": 0,     
        }
    }
    testRVE=RandomInclusionRVE(**initParameters)


    modelingParameters={                                                            
        "placementOptions": {"maxAttempts": 10000,                                  
                             "minRelDistBnd": 0.1,                                  
                             "minRelDistInc": 0.1,                                  
        }
    }
    meshingParameters={                                                             
                "threads": None,                                                            
                "refinementOptions": {"maxMeshSize": size,                                
                                      "inclusionRefinement": False,                          
                                      "interInclusionRefinement": False,                    
                                      "elementsPerCircumference": 10,                       
                                      "elementsBetweenInclusions": 10,                       
                                      "inclusionRefinementWidth": 5,                        
                                      "transitionElements": "auto",                         
                                      "aspectRatio": 1.5}}


    testRVE.createGmshModel(**modelingParameters)
    testRVE.createMesh(**meshingParameters)
    testRVE.saveMesh("A"+str(int(4/size))+".msh2")
    testRVE.close()

hs = [0.1,0.5,1]
seeds = np.arange(1,2)
for h in hs:
    gen(size=h)

#gen = CreateRVEGeometry()
#
#gen(0.4)
