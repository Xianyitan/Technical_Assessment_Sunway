from PIL import Image
import matplotlib.pyplot as plt

def compare_images_side_by_side(image_path1, image_path2, title1='Image 1', title2='Image 2'):
    # Load images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Display the first image
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')  # Hide axes

    # Display the second image
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')  # Hide axes

    # Show the plot
    plt.show()

blurred_image_path = "C:\\Users\\User\\Desktop\\Sunway_Test\\DeblurGANv2\\test_img\\blur.jpg"
deblurred_image_path = "C:\\Users\\User\\Desktop\\Sunway_Test\\DeblurGANv2\\submit\\blur.jpg"

compare_images_side_by_side(blurred_image_path, deblurred_image_path, 'Blurred Image', 'Deblurred Image')