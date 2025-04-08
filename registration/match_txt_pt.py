import os
import re

def find_match(txt_file, pt_dir):
    pt = os.listdir(pt_dir)
    t = txt_file
    matching_file = find_matching_file(t, pt)
    if matching_file:
        return matching_file
    else:
        print(f'No matching file found for {t}')
        exit(-1)



def find_matching_file(target, files):
    def extract_info(filename):
        # Extract relevant parts of the filename based on the known patterns
        patterns = [
            r'([FM])(\d{1,2})_(\d{1,2})_system\d+\.pt',  # F1_03_system1.pt, M1_03_system1.pt
            r'([FM])(\d{1,2})_(\d{1,2})\.pt',            # F1_03.pt, M1_03.pt
            r'system_\d+_target_([FM](\d{1,2})_(\d{1,2}))\.txt'  # system_1_target_F1_03.txt, M1_03.txt
        ]

        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                return f"{match.group(1)}{int(match.group(2))}_{int(match.group(3))}"
        return None

    def normalize_info(info):
        if info is None:
            return None
        # Normalize the extracted info to ensure zero padding for comparison
        match = re.match(r'([FM])(\d{1,2})_(\d{1,2})', info)
        if match:
            return f"{match.group(1)}{int(match.group(2)):02}_{int(match.group(3)):02}"
        return info

    target_info = normalize_info(extract_info(target))
    if target_info is None:
        return None

    for file in files:
        file_info = normalize_info(extract_info(file))
        if target_info == file_info:
            return file

    return None
