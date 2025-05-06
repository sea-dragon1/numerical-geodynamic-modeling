import json

json_str = '''
{
  "version": "0.4",
  "cross section":[[0,0],[4000e3,0]],
  "features":
  [  

    {"model":"continental plate", "name":"LCP", "coordinates":[[0e3,0e3],[200e3,0e3],[200e3,120e3],[0e3,120e3]],
         "temperature models":[{"model":"linear", "max depth":120e3, "bottom temperature":1573}],
         "composition models":[{"model":"uniform", "compositions":[0], "max depth":25e3},
                               {"model":"uniform", "compositions":[1], "min depth":25e3, "max depth":35e3},
                               {"model":"uniform", "compositions":[2], "min depth":35e3, "max depth":120e3}]},
   
    {"model":"oceanic plate", "name":"LOP", "max depth":80e3,
           "coordinates":[[200e3,80e3],[800e3,80e3],[800e3,0e3],[200e3,0e3]],
           "temperature models":[{"model":"linear", "max depth":80e3}],
           "composition models":[{"model":"uniform", "compositions":[3], "max depth":4e3},
                                 {"model":"uniform", "compositions":[4], "min depth":4e3, "max depth":12e3},
                                 {"model":"uniform", "compositions":[5], "min depth":12e3, "max depth":80e3}]},

    {"model":"oceanic plate", "name":"LWZ", "max depth":80e3,
           "coordinates":[[800e3,0e3],[810e3,0e3],[810e3,80e3],[800e3,80e3]],
           "temperature models":[{"model":"linear", "max depth":80e3}],
           "composition models":[{"model":"uniform", "compositions":[6], "max depth":80e3}]},

    {"model":"continental plate", "name":"MCP", "max depth":120e3,
           "coordinates":[[810e3,0e3],[2200e3,0e3],[2200e3,120e3],[810e3,120e3]],
           "temperature models":[{"model":"linear", "max depth":120e3}],
           "composition models":[{"model":"uniform", "compositions":[7], "max depth":25e3},
                                 {"model":"uniform", "compositions":[8], "min depth":25e3, "max depth":35e3},
                                 {"model":"uniform", "compositions":[9], "min depth":35e3, "max depth":120e3}]},

    {"model":"continental plate", "name":"SCOP", "max depth":80e3,
           "coordinates":[[2200e3,0e3],[2600e3,0e3],[2600e3,80e3],[2200e3,80e3]],
           "temperature models":[{"model":"linear", "max depth":80e3}],
           "composition models":[{"model":"uniform", "compositions":[10], "max depth":5e3},
                                 {"model":"uniform", "compositions":[11], "min depth":5e3, "max depth":15e3},
                                 {"model":"uniform", "compositions":[12], "min depth":15e3, "max depth":80e3}]},

    {"model":"oceanic plate", "name":"RWZ", "max depth":80e3,
           "coordinates":[[2600e3,0e3],[2610e3,0e3],[2610e3,80e3],[2600e3,80e3]],
           "temperature models":[{"model":"linear", "max depth":80e3}],
           "composition models":[{"model":"uniform", "compositions":[13], "max depth":80e3}]},

    {"model":"oceanic plate", "name":"ROP", 
            "coordinates":[[2610e3,0e3],[3510e3,0e3],[3510e3,80e3],[2610e3,80e3]],
            "temperature models":[{"model":"linear", "max depth":80e3, "bottom temperature":1573}],
            "composition models":[{"model":"uniform", "compositions":[14], "max depth":4e3},
                                  {"model":"uniform", "compositions":[15], "min depth":4e3, "max depth":12e3},
                                  {"model":"uniform", "compositions":[16], "min depth":12e3, "max depth":80e3}]},

    {"model":"oceanic plate", "name":"RRWZ", "max depth":80e3,
           "coordinates":[[3510e3,0e3],[3520e3,0e3],[3520e3,80e3],[3510e3,80e3]],
           "temperature models":[{"model":"linear", "max depth":80e3}],
           "composition models":[{"model":"uniform", "compositions":[17], "max depth":80e3}]},
    
    {"model":"oceanic plate", "name":"RROP", 
        "coordinates":[[3520e3,0e3],[4000e3,0e3],[4000e3,80e3],[3520e3,80e3]],
        "temperature models":[{"model":"linear", "max depth":80e3, "bottom temperature":1573}],
        "composition models":[{"model":"uniform", "compositions":[18], "max depth":4e3},
                              {"model":"uniform", "compositions":[19], "min depth":4e3, "max depth":12e3},
                              {"model":"uniform", "compositions":[20], "min depth":12e3, "max depth":80e3}]},

    {"model":"subducting plate", "name":"Subducting plate", "coordinates":[[3520e3,-1e3],[3520e3,1e3]], "dip point":[3420e3,0],
         "segments":[{"length":50e3, "thickness":[80e3], "angle":[0,45]}, {"length":100e3, "thickness":[80e3], "angle":[45]}],
         "temperature models":[{"model":"plate model", "density":3370, "plate velocity":0.02, "adiabatic heating":false}],
         "composition models":[{"model":"uniform", "compositions":[18], "max distance slab top":4e3},
                               {"model":"uniform", "compositions":[19], "min distance slab top":4e3, "max distance slab top":12e3},
                               {"model":"uniform", "compositions":[20], "min distance slab top":12e3, "max distance slab top":80e3 }]},

        
    {"model":"continental plate", "name":"PS", "coordinates":[[0e3,0e3],[4000e3,0e3],[4000e3,120e3],[0e3,120e3]],
                 "composition models":[{"model":"uniform", "compositions":[21], "operation":"add", "max depth":120e3}]},

    
    {"model":"mantle layer", "name":"upper mantle", "min depth":120e3, "max depth":660e3, "coordinates":[[0e3,0e3],[4000e3,0e3],[4000e3,660e3],[0e3,660e3]],
         "temperature models":[{"model":"linear", "min depth":85e3, "max depth":660e3, "top temperature":1573, "bottom temperature":1913}],
         "composition models":[{"model":"uniform", "compositions":[22]}]}
  ]
}
'''

try:
    data = json.loads(json_str)
    print("JSON 格式正确")
except json.JSONDecodeError as e:
    print(f"JSON 格式错误: {e}")