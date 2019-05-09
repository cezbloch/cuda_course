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
    cuda_props = "$(SolutionDir)cuda.props"
    tree = El.parse(project)
    root = tree.getroot()
    cuda_props_found = find_attribute(root, cuda_props)
    if not cuda_props_found:
        remove_cuda_target(root)
        tree.write("c:\out.xml")


def remove_cuda_target(root):
    cuda_targets = "$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.targets"
    for child in root:
        if 'Label' in child.attrib:
            label = child.attrib['Label']
            if label == 'ExtensionTargets':
                for import_tag in child:
                    if 'Project' in import_tag.attrib:
                        project_attribute = import_tag.attrib['Project']
                        if project_attribute == cuda_targets:
                            root.remove(child)



def find_attribute(root, at):
    for child in root:
        if 'Label' in child.attrib:
            label = child.attrib['Label']
            if label == 'ProjectConfigurations':
                for import_tag in child:
                    if 'Project' in import_tag.attrib:
                        project_attribute = import_tag.attrib['Project']
                        if project_attribute == at:
                            return True

    return False


def main():
    projects = find_visual_studio_projects()
    for project in projects:
        make_project_cuda_version_independent(project)


if __name__ == "__main__":
    main()
