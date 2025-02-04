import os
import csv
import shutil
import random
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def xGroupedSum(directory):
    # Dictionary to store the counts of y for each x
    grouped_images = defaultdict(set)

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            # Extract x and y from the filename
            x, y = filename.split('_')[0], filename.split('_')[1].split('.')[0]
            grouped_images[x].add(y)

    # Sort the results by the number of y values for each x
    sorted_grouped_images = sorted(
        grouped_images.items(), key=lambda item: len(item[1]), reverse=True)

    # Print the sorted results
    for x, y_set in sorted_grouped_images:
        print(f'{x}: {len(y_set)}')

    # Save the sorted results to a file named x_grouped_sum.txt in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_directory, 'x_grouped_sum.txt')
    with open(output_file, 'w') as f:
        for x, y_set in sorted_grouped_images:
            f.write(f'{x}: {len(y_set)}\n')


def process_image(filepath):
    img = cv2.imread(filepath)
    if img is not None:
        height, width = img.shape[:2]
        return (filepath, width, height)
    return None


def create_heatmap(directory):
    # Dictionary to store the counts of each dimension
    dimension_counts = defaultdict(int)

    # Get the list of image files
    image_files = [os.path.join(directory, filename) for filename in os.listdir(
        directory) if filename.endswith('.jpg')]

    # Use ThreadPoolExecutor to parallelize image processing
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, image_files)
        for result in results:
            if result is not None:
                filepath, width, height = result
                dimension_counts[(width, height)] += 1

    # Extract unique dimensions and their counts
    widths = [dim[0] for dim in dimension_counts.keys()]
    heights = [dim[1] for dim in dimension_counts.keys()]
    counts = [count for count in dimension_counts.values()]

    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        widths, heights, bins=(max(widths), max(heights)), weights=counts)

    # Plot the heatmap
    plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of Images')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Dimension Heatmap')
    plt.show()


def write_image_dimensions(directory):
    # Get the list of image files
    image_files = [os.path.join(directory, filename) for filename in os.listdir(
        directory) if filename.endswith('.jpg')]

    # Open a CSV file to write the dimensions
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_directory, 'image_dimensions.csv')
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Width', 'Height'])

        # Use ThreadPoolExecutor to parallelize image processing
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_image, image_files)
            for result in results:
                if result is not None:
                    filepath, width, height = result
                    csvwriter.writerow(
                        [os.path.basename(filepath), width, height])


def copy_sample_images(csv_file, source_directory, destination_directory):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Dictionary to store filenames grouped by their rounded dimensions
    dimension_samples = defaultdict(list)

    # Read the CSV file and group filenames by their rounded dimensions
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            filename, width, height = row
            width = round(float(width) / 10) * 10
            height = round(float(height) / 10) * 10
            dimension = (width, height)
            dimension_samples[dimension].append(filename)

    # Randomly select up to 10 images for each unique dimension
    selected_samples = {}
    for dimension, filenames in dimension_samples.items():
        selected_samples[dimension] = random.sample(
            filenames, min(10, len(filenames)))

    # Copy the selected sample images to the destination directory
    for dimension, filenames in selected_samples.items():
        for filename in filenames:
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_directory, filename)
            shutil.copyfile(source_path, destination_path)


# Directory containing the images
directory = 'E:/test'
# xGroupedSum(directory)
# create_heatmap(directory)
# write_image_dimensions(directory)

# Copy sample images to the new folder
csv_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'image_dimensions.csv')
destination_directory = 'E:/dimension_samples'
copy_sample_images(csv_file, directory, destination_directory)
