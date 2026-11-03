from pathlib import Path
import os
def correct_fold(pth):
    if os.path.exists(pth):
        for file in pth:
            os.remove(file)
    else:
        print("path not found")