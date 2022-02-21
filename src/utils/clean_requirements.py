import re

unused_req_file = "unused_requirements.txt"
req_file = "requirements_test.txt"
out_file = "requirements2.txt"

with open(req_file) as f:
  regex = "[^= ]*"
  req_lines = f.readlines()
  req_packages = [re.findall(regex, line)[0] for line in req_lines]
    

with open(unused_req_file) as f:
  unused_package_names = []
  for idx, line in enumerate(f):
    if idx == 0:
      continue
    package_name = line.split()[0]
    unused_package_names.append(package_name)
    

filtered_packages = []
for line, package_name in zip(req_lines, req_packages):
  if package_name not in unused_package_names:
    filtered_packages.append(line)

with open(out_file, "w") as f:
  for line in filtered_packages:
    f.write(line)