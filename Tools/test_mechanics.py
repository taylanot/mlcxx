from Mechanics.fenics_.src.genrve import *
from gmshModel.Model import RandomInclusionRVE
import numpy as np
import os 
from copy import deepcopy

def gen(seed=2, size=0.1):
    np.random.seed(seed)
    initParameters={                                                                
        "inclusionSets": [1, 3],                                                   
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
                                      "inclusionRefinement": True,                          
                                      "interInclusionRefinement": False,                    
                                      "elementsPerCircumference": 10,                       
                                      "elementsBetweenInclusions": 10,                       
                                      "inclusionRefinementWidth": 5,                        
                                      "transitionElements": "auto",                         
                                      "aspectRatio": 1.5}}


    testRVE.createGmshModel(**modelingParameters)
    testRVE.createMesh(**meshingParameters)
    testRVE.saveMesh("A"+str(int(4/size))+".msh2")
    testRVE.saveGeometry("A.step")
    testRVE.saveGeometry("A.geo")
    testRVE.close()

gen(size=0.1)
gen(size=0.5)

#gen = CreateRVEGeometry()
#
#gen(0.4)
