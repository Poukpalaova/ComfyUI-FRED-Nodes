import torch
import numpy as np

class FRED_JoinImages():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "images_2": ("IMAGE",),  # Assume this can now be a list of images
                "direction": (["Vertical", "Horizontal"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    
    FUNCTION = "JoinImages"
    
    CATEGORY = "ezXY/image"

    def JoinImages(self, image_1, images_2, direction):
        # Ensure images_2 is a list
        if not isinstance(images_2, list):
            images_2 = [images_2]
        
        # Duplicate image_1 to match the length of images_2
        images_1 = [image_1] * len(images_2)
        
        # Prepare images for plotting
        images = []
        for img1, img2 in zip(images_1, images_2):
            images.append(img1)
            images.append(img2)
        
        # Set x and y positions for plotting
        if direction == "Vertical":
            x = [0] * len(images)
            y = [i for i in range(len(images))]
        elif direction == "Horizontal":
            x = [i for i in range(len(images))]
            y = [0] * len(images)

        # plotter = PlotImages()

        # return plotter.plotXY(images, x, y, False)
        return self.plotXY(images, x, y, False)

    def plotXY(self, images, x_pos, y_pos, force_all = False):
        # make sure we have enough x and y positions
        # if not, repeat the last entry
        if len(x_pos) < len(images):
            x_pos.extend( [x_pos[-1]] * (len(images)-len(x_pos)) );
        if len(y_pos) < len(images):
            y_pos.extend( [y_pos[-1]] * (len(images)-len(y_pos)) );
        
        # find the edges of grid
        column_min, column_max = min(x_pos), max(x_pos)
        row_min, row_max = min(y_pos), max(y_pos)
        
        # size of the grid
        # grid might have more positions than it has input data
        column_length = len(range(column_min, column_max+1))
        row_length = len(range(row_min, row_max+1))
    
        # Check which dimensions need to be padded before concatenation
        pad_dimensions = [0,0]
        if force_all:
            pad_dimensions = [1,1]
        else: 
            if column_length > 1:
                pad_dimensions[0] += 1 
            if row_length > 1:
                pad_dimensions[1] += 1
    
        # create the grid (2d list) of size row_range x column_range items
        # Pretty sweet pattern
        plot = [ [None] * column_length for i in range(row_length) ]
    
        # Capture all image sizes and find the largest values
        max_height = max_width = 0
        image_sizes = []
        for image in images:
            _, _height, _width, _ = image.shape
            max_height = max(max_height, _height)
            max_width = max(max_width, _width)
            image_sizes.append({"height": _height, "width": _width})
    
        # Check if final plot will be too large (in pixels).
        # Change value 81000000 if you want larger images
        pixels = max_height * len(plot) * max_width * len(plot[0])
        if pixels > 81000000:
            message = "ezXY: Plotted image too large\n"
            message = message + f"    Max pixels: {81000000:,}\n"
            message = message + f"    Plot size(approx.) : {pixels:,}\n"
            message = message + "    Returning single image."
            print(message)
            return([images[0]])
    
        # Zero out max height or width if only padding along a single dimension.
        required_height, required_width = np.multiply([max_height,max_width], pad_dimensions)
    
        for i, dims in enumerate(image_sizes):
            # Pad undersized images
            if required_height > dims["height"] or required_width > dims["width"]:
                images[i] = padImage(images[i], (required_height, required_width))
    
            # remap position lists to the new grid's coordinates
            # desired variables look like (0,0) to (column_max, row_max)
            _x, _y = x_pos[i] - column_min, y_pos[i] - row_min
            
            # put each image in it's place
            # index 'i' is synchronised between position, image, and dim lists
            # so everything just kinda works out.
            plot[_y][_x] = images[i]
    
        # I don't know a whole lot about tensors, but this works.
        # Start by iterating through the plot's rows, filling empty positions with blank images
        # Then concatonate the images horizontally, forming rows
        null_image = torch.zeros(1, max_height, max_width, 3)
        for i, row in enumerate(plot):
            for j, item in enumerate(row):
                if not torch.is_tensor(item):
                    row[j] = null_image
            plot[i] = torch.cat(row, 2)
    
        # Finally, concatonate the rows together to form the finished plot
        plot = torch.cat(plot, 1)
        return (plot,)

        def padImage(image, target_dimensions):
            dim0, dim1, dim2, dim3 = image.size()
            _height = max(dim1, target_dimensions[0])
            _width = max(dim2, target_dimensions[1])

            # Blank image of the minimum size
            _image = torch.zeros(dim0, _height, _width, dim3)

            top_pad = (_height - dim1) // 2
            bottom_pad = top_pad + dim1
            
            left_pad = (_width - dim2) // 2
            right_pad = left_pad + dim2
            
            # Very cool image splicing pattern
            # Replaces the center of the blank image with the image from params
            _image[:, top_pad:bottom_pad, left_pad:right_pad, :] = image
            
            return _image

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_JoinImages": FRED_JoinImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_JoinImages": "👑 FRED_JoinImages"
}