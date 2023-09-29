import json
import numpy
import matplotlib.pyplot as plt


reals = []
fakes = []
for i in range(54):
    with open(f"outputs/analysis/real/json/batch_{i}.json") as real_file:
        real_batch = json.load(real_file)
    with open(f"outputs/analysis/fake/json/batch_{i}.json") as fake_file:
        fake_batch = json.load(fake_file)
    
    reals.append(real_batch)
    fakes.append(fake_batch)
hr = [d["hr"] for d in reals]
x = numpy.arange(0, len(hr), step=1)
plt.hist(hr)
plt.savefig("hr.png")
plt.close()
quit()

final_dict = {    
    "real": 
        {
            "hr": sum(d["hr"] for d in reals) / len(reals),
            "hr_std": sum(d["hr_std"] for d in reals) / len(reals),
            
            "sdnn": sum(d["sdnn"] for d in reals) / len(reals),
            "sdnn_std": sum(d["sdnn_std"] for d in reals) / len(reals),
            
            "rmssd": sum(d["rmssd"] for d in reals) / len(reals),
            "rmssd_std": sum(d["rmssd_std"] for d in reals) / len(reals),
            
            "P": sum(d["P"] for d in reals) / len(reals),
            "P_std": sum(d["P_std"] for d in reals) / len(reals),
            
            "LF": sum(d["LF"] for d in reals) / len(reals),
            "LF_std": sum(d["LF_std"] for d in reals) / len(reals),
            
            "HF": sum(d["HF"] for d in reals) / len(reals),
            "HF_std": sum(d["HF_std"] for d in reals) / len(reals),
            
            "VLF": sum(d["VLF"] for d in reals) / len(reals),
            "VLF_std": sum(d["VLF_std"] for d in reals) / len(reals),
        },
    "fake": 
        {
            "hr": sum(d["hr"] for d in fakes) / len(fakes),
            "hr_std": sum(d["hr_std"] for d in fakes) / len(fakes),
            
            "sdnn": sum(d["sdnn"] for d in fakes) / len(fakes),
            "sdnn_std": sum(d["sdnn_std"] for d in fakes) / len(fakes),
            
            "rmssd": sum(d["rmssd"] for d in fakes) / len(fakes),
            "rmssd_std": sum(d["rmssd_std"] for d in fakes) / len(fakes),
            
            "P": sum(d["P"] for d in fakes) / len(fakes),
            "P_std": sum(d["P_std"] for d in fakes) / len(fakes),
            
            "LF": sum(d["LF"] for d in fakes) / len(fakes),
            "LF_std": sum(d["LF_std"] for d in fakes) / len(fakes),
            
            "HF": sum(d["HF"] for d in fakes) / len(fakes),
            "HF_std": sum(d["HF_std"] for d in fakes) / len(fakes),
            
            "VLF": sum(d["VLF"] for d in fakes) / len(fakes),
            "VLF_std": sum(d["VLF_std"] for d in fakes) / len(fakes),
        }
}

# with open("outputs/analysis/total.json", "x") as json_file:
#     json.dump(final_dict, json_file, indent=6)