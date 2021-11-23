from Mechanics.fenics_.src.genrve import *
#from gmshModel.Model import RandomInclusionRVE
#import numpy as np
#import os 
#
#
#initParameters={                                                                
#    "inclusionSets": [1, 3],                                                   
#    "inclusionType": "Circle",                                                  
#    "size": [4, 4, 0],                                                          
#    "origin": [0, 0, 0],                                                        
#    "periodicityFlags": [1, 1, 1],                                              
#    "domainGroup": "domain",                                                    
#    "inclusionGroup": "inclusions",                                             
#    "gmshConfigChanges": {"General.Terminal": 0,                                
#                          "Mesh.CharacteristicLengthExtendFromBoundary": 0,     
#    }
#}
#testRVE=RandomInclusionRVE(**initParameters)
#
#
#modelingParameters={                                                            
#    "placementOptions": {"maxAttempts": 10000,                                  
#                         "minRelDistBnd": 0.1,                                  
#                         "minRelDistInc": 0.1,                                  
#    }
#}
#testRVE.createGmshModel(**modelingParameters)
#
#
#testRVE.createMesh(**meshingParameters)
#
#testRVE.saveGeometry("A.step")
#testRVE.close()

gen = CreateRVEGeometry()

gen(0.4)
