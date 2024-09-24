# import os
import importlib

# #uploadimg
# def modify_js_file(file_path, new_content):
    # with open(file_path, 'r', encoding='utf-8') as file:  # Read the current content of the file
        # content = file.read()

    # # Now, use `new_content` to check if modifications are necessary
    # if "image_upload_dress" not in content:

        # insert_position = content.find('nodeData.input.required.upload = ["IMAGEUPLOAD"];')
        # if insert_position != -1:

            # insert_position += len('nodeData.input.required.upload = ["IMAGEUPLOAD"];')
            # # Here, you should modify `content` based on `new_content`, before writing it back
            # content = content[:insert_position] + new_content + content[insert_position:]
            # with open(file_path, 'w', encoding='utf-8') as file:  # Write the modified content back to the file
                # file.write(content)
            # print(f"File '{file_path}' updated successfully.✅")
        # else:
            # print("Original code block not found.❌")
    # else:
        # print("File already contains the necessary modifications.✅")

# new_js_content = """
        # }
        # if (nodeData?.input?.required?.image?.[1]?.image_upload_dress === true) {
            # nodeData.input.required.upload = ["DRESS_IMAGEUPLOAD"];
        # }
        # if (nodeData?.input?.required?.image?.[1]?.image_upload_hair_style === true) {
            # nodeData.input.required.upload = ["HAIR_STYLE_IMAGEUPLOAD"];
        # }
        # if (nodeData?.input?.required?.image?.[1]?.image_upload_eyes_color === true) {
            # nodeData.input.required.upload = ["EYES_COLOR_IMAGEUPLOAD"];
        # }
        # if (nodeData?.input?.required?.image?.[1]?.image_upload_top === true) {
            # nodeData.input.required.upload = ["TOP_IMAGEUPLOAD"];
        # }
        # if (nodeData?.input?.required?.image?.[1]?.image_upload_hair_color === true) {
            # nodeData.input.required.upload = ["HAIR_COLOR_IMAGEUPLOAD"];

# """

# current_dir = os.path.dirname(os.path.abspath(__file__))
# uploadimg_js_file_path = os.path.join(current_dir, '../../web/extensions/core/uploadImage.js')
# print(uploadimg_js_file_path)

# modify_js_file(uploadimg_js_file_path, new_js_content)

# #folderpath
# def modify_py_file(file_path, new_content, search_line, function_content, search_function):
    # with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding here
        # lines = file.readlines()

    # new_content_key_line = new_content.strip().split('\n')[0]
    # function_content_key_line = function_content.strip().split('\n')[0]

    # if new_content_key_line not in "".join(lines):
        # for index, line in enumerate(lines):
            # if search_line in line:
                # lines.insert(index + 1, new_content)
                # break
    # if function_content_key_line not in "".join(lines):
        # function_start = False
        # for index, line in enumerate(lines):
            # if search_function in line:
                # function_start = True
            # if function_start and "return None" in line:
                # lines.insert(index, function_content)
                # break
    # with open(file_path, 'w', encoding='utf-8') as file:  # Specify encoding here
        # file.writelines(lines)
    # print(f"File '{file_path}' updated successfully.✅")

# new_py_content = """
# supported_images_extensions = [".jpg", ".png", ".jpeg"]
# dress_dir = os.path.join(base_path, "custom_nodes", "ComfyUI-FRED-Nodes", "img_lists", "dress")
# folder_names_and_paths["dress"] = ([dress_dir], supported_images_extensions)

# hair_style_dir = os.path.join(base_path, "custom_nodes", "ComfyUI-FRED-Nodes", "img_lists", "hair_style")
# folder_names_and_paths["hair_style"] = ([hair_style_dir], supported_images_extensions)

# eyes_color_dir = os.path.join(base_path, "custom_nodes", "ComfyUI-FRED-Nodes", "img_lists", "eyes_color")
# folder_names_and_paths["eyes_color"] = ([eyes_color_dir], supported_images_extensions)

# top_dir = os.path.join(base_path, "custom_nodes", "ComfyUI-FRED-Nodes", "img_lists", "top")
# folder_names_and_paths["top"] = ([top_dir], supported_images_extensions)

# hair_color_dir = os.path.join(base_path, "custom_nodes", "ComfyUI-FRED-Nodes", "img_lists", "hair_color")
# folder_names_and_paths["hair_color"] = ([hair_color_dir], supported_images_extensions)
# """

# function_py_content = '''\
    # if type_name == "dress":
        # return folder_names_and_paths["dress"][0][0]
    # if type_name == "hair_style":
        # return folder_names_and_paths["hair_style"][0][0]
    # if type_name == "eyes_color":
        # return folder_names_and_paths["eyes_color"][0][0]
    # if type_name == "top":
        # return folder_names_and_paths["top"][0][0]
    # if type_name == "hair_color":
        # return folder_names_and_paths["hair_color"][0][0]

# '''

# py_file_path = os.path.join(current_dir, '../../folder_paths.py')

# modify_py_file(py_file_path, new_py_content, 'folder_names_and_paths["classifiers"]', function_py_content, 'def get_directory_by_type(type_name):')


# #wedget
# def modify_wedgets_js_file(file_path, new_content, new_content_2):
    # with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding here
        # content = file.read()
    # if "DRESS_IMAGEUPLOAD" not in content:
        # insert_position = content.find('return (display==="slider") ? "slider" : "number"')
        # if insert_position != -1:
            # insert_position += len('return (display==="slider") ? "slider" : "number"')
            # content = content[:insert_position] + new_content + content[insert_position:]

        # insert_position_2 = content.find('return { widget: uploadWidget };')
        # if insert_position_2 != -1:
            # insert_position_2 += len('return { widget: uploadWidget };')
            # content = content[:insert_position_2] + new_content_2 + content[insert_position_2:]
            # with open(file_path, 'w', encoding='utf-8') as file:  # Specify encoding here
                # file.write(content)
            # print(f"File '{file_path}' updated successfully.✅")
        # else:
            # print("Original code block not found.❌")
    # else:
        # print("File already contains the necessary modifications.✅")


# new_wedgets_js_content = """
# }
# function createImageUploadWidget(node, inputName, inputData, imageType, app) {
    # const imageWidget = node.widgets.find((w) => w.name === (inputData[1]?.widget ?? "image"));
    # let AuploadWidget;

    # function showImage(name, type) {
        # const img = new Image();
        # img.onload = () => {
            # node.imgs = [img];
            # app.graph.setDirtyCanvas(true);
        # };
        # let folder_separator = name.lastIndexOf("/");
        # let subfolder = "";
        # if (folder_separator > -1) {
            # subfolder = name.substring(0, folder_separator);
            # name = name.substring(folder_separator + 1);
        # }
        # img.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=${type}&subfolder=${subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
        # node.setSizeForImage?.();
    # }

    # var default_value = imageWidget.value;
    # Object.defineProperty(imageWidget, "value", {
        # set: function (value) {
            # this._real_value = value;
        # },

        # get: function () {
            # let value = "";
            # if (this._real_value) {
                # value = this._real_value;
            # } else {
                # return default_value;
            # }

            # if (value.filename) {
                # let real_value = value;
                # value = "";
                # if (real_value.subfolder) {
                    # value = real_value.subfolder + "/";
                # }

                # value += real_value.filename;

                # if (real_value.type && real_value.type !== "input")
                    # value += ` [${real_value.type}]`;
            # }
            # return value;
        # }
    # });
    # const cb = node.callback;
    # imageWidget.callback = function () {
        # showImage(imageWidget.value, imageType);
        # if (cb) {
            # return cb.apply(this, arguments);
        # }
    # };
    # requestAnimationFrame(() => {
        # if (imageWidget.value) {
            # showImage(imageWidget.value, imageType);
        # }
    # });

    # return { widget: AuploadWidget };
# """
# new_wedgets_js_content_2 = """
    # },
    # DRESS_IMAGEUPLOAD(node, inputName, inputData, app) {
        # return createImageUploadWidget(node, inputName, inputData, 'dress', app);
    # },
    # HAIR_STYLE_IMAGEUPLOAD(node, inputName, inputData, app) {
        # return createImageUploadWidget(node, inputName, inputData, 'hair_style', app);
    # },
    # EYES_COLOR_IMAGEUPLOAD(node, inputName, inputData, app) {
        # return createImageUploadWidget(node, inputName, inputData, 'eyes_color', app);
    # },
    # TOP_IMAGEUPLOAD(node, inputName, inputData, app) {
        # return createImageUploadWidget(node, inputName, inputData, 'top', app);
    # },
    # HAIR_COLOR_IMAGEUPLOAD(node, inputName, inputData, app) {
        # return createImageUploadWidget(node, inputName, inputData, 'hair_color', app);
# """

# wedgets_js_file_path = os.path.join(current_dir, '../../web/scripts/widgets.js')

# modify_wedgets_js_file(wedgets_js_file_path, new_wedgets_js_content, new_wedgets_js_content_2)

# from .FRED_ImageBrowser import NODE_CLASS_MAPPINGS as FRED_ImageBrowser_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FRED_ImageBrowser_DISPLAY
# from .FRED_CropFace import NODE_CLASS_MAPPINGS as FRED_CropFace_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FRED_CropFace_DISPLAY
# from .FRED_LoadImage_V2 import NODE_CLASS_MAPPINGS as FRED_LoadImage_V2_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FRED_LoadImage_V2_DISPLAY
# from .FRED_LoadImage_V3 import NODE_CLASS_MAPPINGS as FRED_LoadImage_V3_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FRED_LoadImage_V3_DISPLAY
# from .FRED_AutoCropImage_SDXL_Ratio_v3 import NODE_CLASS_MAPPINGS as FRED_Auto_Crop_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FRED_Auto_Crop_DISPLAY

# __all__ = [
    # 'FRED_ImageBrowser_CLASS', 'FRED_ImageBrowser_DISPLAY',
    # 'FRED_CropFace_CLASS', 'FRED_CropFace_DISPLAY',
    # 'FRED_LoadImage_V2_CLASS', 'FRED_LoadImage_V2_DISPLAY',
    # 'FRED_LoadImage_V3_CLASS', 'FRED_LoadImage_V3_DISPLAY',
    # 'FRED_Auto_Crop_CLASS', 'FRED_Auto_Crop_DISPLAY',
# ]

# node_list = [
    # # "fred_auto_crop_image_sdxl_ratio_v2",
    # "FRED_AutoCropImage_SDXL_Ratio_v3.py",
    # "FRED_CropFace.py",
    # "FRED_LoadImage_V2.py",
    # "FRED_LoadImage_V3.py",
    # "FRED_ImageBrowser.py"
# ]

# node_class_mappings = {}
# node_display_name_mappings = {}

# for module_name in node_list:
    # print(f"Importing module: {module_name}")
    # imported_module = importlib.import_module(module_name, package=__name__)
    # if not hasattr(imported_module, 'node_class_mappings'):
        # print(f"Module {module_name} lacks NODE_CLASS_MAPPINGS")
        # continue
    # node_class_mappings = {**node_class_mappings, **imported_module.node_class_mappings}
    # node_display_name_mappings = {**node_display_name_mappings, **imported_module.node_display_name_mappings}

# __all__ = ['node_class_mappings', 'node_display_name_mappings']

# from .FRED_ImageBrowser import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# from .FRED_CropFace import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# from .FRED_LoadImage_V2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# from .FRED_LoadImage_V3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# from .FRED_AutoCropImage_SDXL_Ratio_v3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# from .FRED_ImageBrowser import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from .FRED_CropFace import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from .FRED_LoadImage_V2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from .FRED_LoadImage_V3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from .FRED_AutoCropImage_SDXL_Ratio_v3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = [
    # 'NODE_CLASS_MAPPINGS',
    # 'NODE_DISPLAY_NAME_MAPPINGS'
# ]

# def generate_node_mappings(node_config):
    # node_class_mappings = {}
    # node_display_name_mappings = {}

    # for node_name, node_info in node_config.items():
        # node_class_mappings[node_name] = node_info["class"]
        # node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    # return node_class_mappings, node_display_name_mappings

# NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


# node_list = [
    # "FRED_ImageBrowser",
    # "FRED_CropFace",
    # "FRED_LoadImage_V2",
    # "FRED_LoadImage_V3",
    # "FRED_AutoCropImage_SDXL_Ratio_v3"
# ]

# NODE_CLASS_MAPPINGS = {}
# NODE_DISPLAY_NAME_MAPPINGS = {}

# for module_name in node_list:
    # imported_module = importlib.import_module(".py.{}".format(module_name), __name__)
    # NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    # NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


# node_list = [
    # "FRED_ImageBrowser",
    # "FRED_CropFace",
    # "FRED_LoadImage_V2",
    # "FRED_LoadImage_V3",
    # "FRED_AutoCropImage_SDXL_Ratio_v3"
# ]

# NODE_CLASS_MAPPINGS = {}
# NODE_DISPLAY_NAME_MAPPINGS = {}

# for module_name in node_list:
    # imported_module = importlib.import_module("PY.{}".format(module_name), __name__)
    # NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    # NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

import sys
# print(sys.path)
# sys.path.append('M:/AIgenerated/StableDiffusion/ComfyUI_TEST/custom_nodes')

node_list = [
    "FRED_ImageBrowser",
    "FRED_CropFace",
    "FRED_LoadImage_V2",
    "FRED_LoadImage_V3",
    "FRED_LoadImage_V4",
    "FRED_LoadPathImagesPreivew",
    "FRED_LoadPathImagesPreivew_v2",
    "FRED_AutoCropImage_SDXL_Ratio_v3", 
    "FRED_JoinImages_v1", 
    "FRED_LoadRetinaFace"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    # imported_module = importlib.import_module(f".py.{module_name}", package=__name__)
    # imported_module = importlib.import_module("py.{}".format(module_name), __name__)
    imported_module = importlib.import_module(".{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']