import os
import ifcopenshell, ifcopenshell.api as api
os.makedirs('data/exports', exist_ok=True)
m = api.run('project.create_file', version='IFC4')
m.write('data/exports/_ifc_test.ifc')
print('IFC OK')
