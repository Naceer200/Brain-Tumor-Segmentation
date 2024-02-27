import os, glob
import numpy as np
from scipy import ndimage as ndi
import imageio
from scipy.ndimage import center_of_mass, zoom, measurements,binary_dilation
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label, regionprops
import plotly.graph_objects as go


datasets=imageio.volread("C:/Users/HI/PycharmProjects/pythonProject2/kidneymasks/KidneyD",format='dcm')

filt_m=ndi.median_filter(datasets,size=10)
filt_g=ndi.gaussian_filter(filt_m,sigma=2)
# Choose an appropriate threshold value based on the image intensity range
#hist=ndi.histogram(filt_g,min=0,max=65535,bins=65536)
#print(hist.shape)
#plt.plot(hist)
#plt.show()
# real = 60
threshold = 60 #just as needed
# Apply thresholding to create a binary mask
mask = filt_g > threshold
# Optional: Use morphological operations to refine the mask
# For example, to remove small isolated regions:

def delete(masked):
    new_mask = masked.copy()
    labels = label(masked, background=0)
    idxs = np.unique(labels)[1:]
    COM_xs = np.array([center_of_mass(labels==i)[1] for i in idxs])
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y, COM_x in zip(idxs, COM_ys, COM_xs):
        if (COM_y < 0.60*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_y > 0.70*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x > 0.65*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x > 0.5*masked.shape[0]):
            new_mask[labels==idx] = 1
        elif (COM_x < 0.2*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x < 0.4*masked.shape[0]):
            new_mask[labels==idx] = 1
        else:
            new_mask[labels==idx] = 0
    return new_mask

#nmask = mask(datasets)
#new_mask = delete(datasets)
masks = np.vectorize(clear_border, signature='(n,m)->(n,m)')(mask)
maskss = np.vectorize(ndi.binary_opening, signature='(n,m)->(n,m)')(masks)
masksss = np.vectorize(ndi.binary_closing, signature='(n,m)->(n,m)')(maskss)

#new_masks = ndi.binary_fill_holes(new_mask)
new_mask = np.vectorize(delete, signature='(n,m)->(n,m)')(masksss)
nmask = np.vectorize(ndi.binary_fill_holes, signature='(n,m)->(n,m)')(new_mask)
masked = binary_dilation(nmask, iterations=5)
#masked = np.stack(maskeds, axis=0)

output_folder = "C:/Users/HI/PycharmProjects/pythonProject2/kidney masks"  # Replace with your desired folder path

def save_segmented_slices(images, output_folder='masks'):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the slices and save them as PNG files
    for i, image in enumerate(images):
        # You can customize the file name pattern if needed
        filename = os.path.join(output_folder, f"segmented_slice_{i+1}.png")

        # Save the image using Matplotlib
        plt.imshow(image, cmap='gray')  # Assuming grayscale images, adjust cmap if needed
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close()

def main():
    # Assuming you have a NumPy array of segmented and masked slices
    # Replace this with your actual data
    # Replace with your actual image data

    # Save the segmented slices as PNG files
    save_segmented_slices(masked)

if __name__ == "__main__":
    main()

im = zoom(1*(masked), (0.3,0.3,0.3))
z, y, x = [np.arange(i) for i in im.shape]
z*=4
X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=np.transpose(im,(1,2,0)).flatten(),
    isomin=0.1,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.write_html("test.html")

fig,axes=plt.subplots(nrows=2,ncols=3)
# Display the original image
axes[0,0].imshow(datasets[57],cmap='gray')
axes[0,0].set_title('slice 57')
axes[0,0].axis('off')
axes[0,1].imshow(datasets[62],cmap='gray')
axes[0,1].set_title('slice 62')
axes[0,1].axis('off')
axes[0,2].imshow(datasets[69],cmap='gray')
axes[0,2].set_title('slice 69')
axes[0,2].axis('off')
axes[1,0].imshow(masked[57],cmap='gray')
axes[1,0].set_title('mask 57')
axes[1,0].axis('off')
axes[1,1].imshow(masked[62],cmap='gray')
axes[1,1].set_title('mask 62')
axes[1,1].axis('off')
axes[1,2].imshow(masked[69],cmap='gray')
axes[1,2].set_title('mask 69')
axes[1,2].axis('off')
#plt.imshow(masked[61], cmap="gray")
#plt.colorbar()
plt.show()

