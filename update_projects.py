import xml.etree.ElementTree as El


def find_project_files():
    import glob, os
    os.chdir("./")
    for file in glob.glob("*.vcxproj"):
        print(file)


def find_visual_studio_projects():
    import os
    import pathlib
    vcxproj_paths = list()
    for path, _, files in os.walk("./"):
        for name in files:
            if name.endswith(".vcxproj"):
                vcx_path = pathlib.PurePath(path, name)
                vcxproj_paths.append(vcx_path)
                print(vcx_path)

    return vcxproj_paths


def make_project_cuda_version_independent(project):
    root = El.parse(project).getroot()
    for type_tag in root.findall('ImportGroup/type'):
        value = type_tag.get('foobar')
        print(value)

def main():

    #find_project_files()
    projects = find_visual_studio_projects()
    for project in projects:
        make_project_cuda_version_independent(project)


if __name__ == "__main__":
    main()
