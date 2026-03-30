import os
import glob

folder = r"d:\12 lead ECG\WFDB_ChapmanShaoxing"
diagnoses = set()

for file in glob.glob(os.path.join(folder, "*.hea")):
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("#Dx:"):
                #Dx: 164889003,59118001
                dx_str = line.replace("#Dx:", "").strip()
                codes = dx_str.split(",")
                for code in codes:
                    diagnoses.add(code.strip())

print(diagnoses)
print(len(diagnoses), "unique diagnoses found")
