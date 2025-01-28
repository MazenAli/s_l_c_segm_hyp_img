from tensorflow.keras.models import load_model
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import patchify
import os
from utils_for_data_handling import *


def VISUALISE_SEGMENTATION_RESULTS(IMAGE, FONTSIZE, FIGURE_TITLE, FIGSIZE, CHANNEL,\
                                   DATA, SEGMENTED_IMAGE, CONFIDENCE_PROBABILITY_SPATIAL_DOMAIN, LINES, SAMPLES,
                                   OUTPUT_PATH):



    NUMBER_OF_PIXELS_PREDICTED_AS_SEA=np.sum(SEGMENTED_IMAGE==0)
    NUMBER_OF_PIXELS_PREDICTED_AS_LAND=np.sum(SEGMENTED_IMAGE==1)
    NUMBER_OF_PIXELS_PREDICTED_AS_CLOUDS=np.sum(SEGMENTED_IMAGE==2)

    NUMBER_OF_PIXELS_IN_ONE_IMAGE=LINES*SAMPLES
    SEA_COVERAGE_LEVEL=NUMBER_OF_PIXELS_PREDICTED_AS_SEA/NUMBER_OF_PIXELS_IN_ONE_IMAGE
    LAND_COVERAGE_LEVEL=NUMBER_OF_PIXELS_PREDICTED_AS_LAND/NUMBER_OF_PIXELS_IN_ONE_IMAGE
    CLOUD_COVERAGE_LEVEL=NUMBER_OF_PIXELS_PREDICTED_AS_CLOUDS/NUMBER_OF_PIXELS_IN_ONE_IMAGE



    fig, axes=plt.subplots(1, 4, figsize=FIGSIZE)


    CHANNEL_GRAY_SCALE = DATA[IMAGE, :, :, [CHANNEL]]
    CHANNEL_GRAY_SCALE = np.squeeze(CHANNEL_GRAY_SCALE, axis=0)
    image_gray_scale=axes[0].imshow(CHANNEL_GRAY_SCALE, cmap='gray')
    axes[0].set_title('Gray-scale channel', fontsize=FONTSIZE, fontweight='bold')
    axes[0].set_xlabel('SAMPLES', fontsize=FONTSIZE)
    axes[0].set_ylabel('LINES', fontsize=FONTSIZE)


    colors_for_classes=['#0000C4', '#C47F00', '#6D6D6D'] # Color codes used when labeling the Sea-Land-Cloud-Labeled dataset from HYPSO-1
    cmap_for_classes=ListedColormap(colors_for_classes)

    segmented_image=axes[1].imshow(SEGMENTED_IMAGE, cmap=cmap_for_classes)
    axes[1].set_title('Segmented image', fontsize=FONTSIZE, fontweight='bold')
    axes[1].set_xlabel('SAMPLES', fontsize=FONTSIZE)
    axes[1].set_ylabel('LINES', fontsize=FONTSIZE)


    confindence_probability_distribution_fig = axes[2].imshow(CONFIDENCE_PROBABILITY_SPATIAL_DOMAIN, cmap='hot', interpolation='nearest')
    axes[2].set_title('Confidence probability', fontsize=FONTSIZE, fontweight='bold')
    axes[2].set_xlabel('SAMPLES', fontsize=FONTSIZE)
    axes[2].set_ylabel('LINES', fontsize=FONTSIZE)


    cbar0=fig.colorbar(image_gray_scale, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.15)

    cbar1=fig.colorbar(segmented_image, ax=axes[1], ticks=[0, 1, 2], orientation='horizontal', fraction=0.046, pad=0.15)
    tick_classes={0: 'Sea', 1: 'Land', 2: 'Cloud'}
    cbar1.set_ticklabels([tick_classes[a_tick] for a_tick in cbar1.get_ticks()])

    cbar2=fig.colorbar(confindence_probability_distribution_fig, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.15)


    for label in cbar0.ax.get_xticklabels():
        label.set_fontsize(FONTSIZE)

    for label in cbar1.ax.get_xticklabels():
        label.set_fontsize(FONTSIZE)

    for label in cbar2.ax.get_xticklabels():
        label.set_fontsize(FONTSIZE)



    coverage_distribution = [SEA_COVERAGE_LEVEL, LAND_COVERAGE_LEVEL, CLOUD_COVERAGE_LEVEL]
    CLASSES_NAMES=['Sea', 'Land', 'Clouds']
    axes[3].pie(coverage_distribution,\
                labels=CLASSES_NAMES, colors=colors_for_classes,
                autopct=lambda p: f'{p:.2f}%', textprops={'fontweight': 'bold', 'fontsize': FONTSIZE, 'color': 'white'},
                startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    axes[3].axis('equal')


    for text in axes[3].texts:
        if text.get_text() in CLASSES_NAMES:
            text.set_color('black')

    plt.suptitle(FIGURE_TITLE, fontsize=FONTSIZE, fontweight='bold')
    OUTPUT = os.path.join(OUTPUT_PATH, f"imagenum_{IMAGE}.png")
    plt.savefig(OUTPUT, bbox_inches='tight', dpi=300)
    plt.close(fig)

def try_dims():
    file = "./data/5-20220905_CaptureDL_00_fagradalsfjall_t_l_2022_09_05T12_37_23-radiance.npy"
    array = np.load(file)
    print(array.shape)
    print(array)
    print("==========================")
    print("==========================")
    print("==========================")
    file = "./data/fagradalsfjall_t_l_2022_09_05T12_37_23_class_NPY_FORMAT.npy"
    array = np.load(file)
    print(array.shape)
    print(array)

def main():
    utils_o=utils_for_data_handling() # Create object of local class

    ### DATA ###

    ### --> FOR USER: Enter next the local path where you have stored the images. Make sure to finish the path by /
    DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED = './data/'  # Example path

    PATHS_TO_DATA = []  # Empty list for paths
    # The images below are directly accessible for download in the supplementary material of our article: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning"
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '5-20220905_CaptureDL_00_fagradalsfjall_t_l_2022_09_05T12_37_23-radiance.npy')
    """
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '8-20220911_CaptureDL_00_haida_2022_09_11T19_24_10-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '22-20221118_CaptureDL_sanrafael_2022_11_16T14_13_57-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '23-20221011_CaptureDL_rrvPlaya_2022_10_10T18_22_33-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '33-20221215_CaptureDL_maunaloa_2022_12_02T20_03_26-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '55-20221209_CaptureDL_bohai_2022_12_07T03_02_01-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '66-20221203_CaptureDL_falklands_2022_12_02T12_27_05-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '75-20221130_CaptureDL_vancouver_2022_11_27T19_22_56-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '78-20221130_CaptureDL_florida_2022_11_25T15_10_03-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '87-20221126_CaptureDL_tenerife_2022_11_21T11_15_03-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '88-20221126_CaptureDL_qatar_2022_11_20T06_42_58-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '91-20221126_CaptureDL_jiaozhouBayBridge_2022_11_20T01_54_58-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '92-20221126_CaptureDL_trondheim_2022_11_19T09_56_00-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '117-20221023_CaptureDL_gulfOfAqaba_07_49_08-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '122-20221020_CaptureDL_gobabeb_2022_10_20T08_39_51-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '135-20221010_CaptureDL_patra_2022_10_10T08_51_44-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '138-20221010_CaptureDL_tampa_target_l_2022_10_07T15_51_50-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '144-20220928_CaptureDL_00_belem_2022_09_28T13_03_32-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '145-20220927_CaptureDL_00_finnmark_71.1958_26.1931_2022_09_27T09_45_45-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '147-20220925_CaptureDL_00_santarem_2022_09_25T13_40_04-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '149-20220923_CaptureDL_00_lisbon_2022_09_23T10_43_19-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '176-20220908_CaptureDL_00_fagradalsfjall_2022_09_08T12_01_47-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '181-20220905_CaptureDL_00_eerie_t_r_2022_09_05T15_53_52-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '186-20220829_CaptureDL_00_runde_2022_08_29T10_50_39-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '208-20220806_CaptureDL_00_vancouverisland_2022_08_06T18_37_57-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '212-20220805_CaptureDL_00_balaton_2022_08_05T09_18_56-radiance.npy')
    """

    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '262-20220630_CaptureDL_00_grieg_T1929-radiance.npy')

    """
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '268-20220621_Capture_DL_00_mjosa_T10_05-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '272-20220610_CaptureDL_00_Lofoten_10_34-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '274-20230215_CaptureDL_yangon_2023-02-15_0330Z-radiance.npy')
    PATHS_TO_DATA.append(DIRECTORY_WHERE_DATA_FOR_INFERENCE_IS_STORED + '275-20230313_CaptureDL_chapala_2023-02-19_1649Z_2-radiance.npy')
    """

    NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET =len(PATHS_TO_DATA)
    print('Number of paths to images: ', NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET)


    ### MODEL ###

    DATA_PROCESSING_MODE = '1D_PROCESSING'  # Choose: '1D_PROCESSING', or '3D_PROCESSING'

    if DATA_PROCESSING_MODE == '1D_PROCESSING':
        # As an example, we use the model 'DEEP_LEARNING_NN_classifier_20231009_203324'
        #   The model file corrsponds to 1D-Justo-LiuNet trained on L1b calibrated radiance for 112 channels
        #   (NB: Instead of using the 120 channels in the dataset, we exclude channels 0, 1, 2, 3, 106, 107, 108, 109
        #           - details provided in the article referenced at the start of this Python notebook)

        # Example path - models available in supplementary material of article cited at the start of this notebook
        PATH_TO_MODEL = './models/DEEP_LEARNING_NN_classifier_20231009_203324'

        # Note: When testing different models, make sure to feed the models with data of the same characteristics as in the training, as explained next.
        #            For instance, if a model has been trained on L1b radiance, then the data during inference must also be L1b radiance - likewise for unprocessed raw data.
        #            In the pre-processing stages, we remove certain channels. Make sure to pick the 'right' channels to remove, as illustrated next.
        #            For instance, if a model has been trained with 112 channels, then it must receive during inference 112 channels and not 3 for example.

    elif DATA_PROCESSING_MODE == '3D_PROCESSING':
        # As an example, we use the model 'DEEP_LEARNING_NN_classifier_20231009_140540'
        #   The model file corrsponds to 2D-CUNet++ Reduced trained on L1b calibrated radiance for patch size 48 x 48 and 112 channels
        #   (NB: Instead of using the 120 channels in the dataset, we exclude channels 0, 1, 2, 3, 106, 107, 108, 109
        #           - details provided in the article referenced at the start of this Python notebook)

        # Example path - models available in supplementary material of article cited at the start of this notebook
        PATH_TO_MODEL = './models/DEEP_LEARNING_NN_classifier_20231009_140540'

    print('Loading model...')
    model_classifier = load_model(PATH_TO_MODEL)
    print('Model loaded ok! See architectural details next. \n')
    model_classifier.summary()

    ### LOAD IMAGES ###

    LINES = 956
    SAMPLES = 684
    CHANNELS = 120  # Lines (frames), slit samples (swath), and channels are set to default values for HYPSO-1
    DATA = -1 * np.ones((NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, LINES, SAMPLES, CHANNELS),
                        dtype=np.float32)  # Radiance is float32 (4 bytes are enough)

    print('Loading data...')
    for iterator_images in range(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET):
        DATA[iterator_images, :, :, :] = np.load(PATHS_TO_DATA[iterator_images])

    print('Loaded! The data is loaded in an array of dimensions: ', DATA.shape)

    ### SAVE IMAGE EXAMPLE ###

    # Choose one image in the array DATA. As an example, we pick next channel 89 which corresponds to the visible red (699.61 nm) in L1b calibration
    IMAGE = 0
    CHANNEL = 89

    CHANNEL_GRAY_SCALE = DATA[IMAGE, :, :, [CHANNEL]]  # Result has dimensions: 1 x 956 x684
    CHANNEL_GRAY_SCALE = np.squeeze(CHANNEL_GRAY_SCALE,
                                    axis=0)  # Result has dimensions: 956 x684 (remove first dimension)

    print('The channel from the image has dimensions: ', CHANNEL_GRAY_SCALE.shape,
          '. Visualisation below (gray scale).')
    normalized_image = CHANNEL_GRAY_SCALE / CHANNEL_GRAY_SCALE.max()
    plt.xlabel('SAMPLES (SWATH)')
    plt.ylabel('LINES (FRAMES)')
    output_file = os.path.join("./images", "channel_gray_scale_image.png")
    plt.imsave(output_file, normalized_image, cmap='gray')
    print(f"Image saved as {output_file}.")

    ### REMOVE CHANNELS ###

    CHANNELS_TO_REMOVE = [0, 1, 2, 3, 106, 107, 108, 109]
    # In this notebook, we employ 112 out of the 120 channels of the dataset. Further reasons in our article
    # referenced at the start of this notebook.
    # NB: The channel numbers refer to L1b calibration.

    print('Dimensions of data before removing channels: ', DATA.shape)
    print('Removing channels...')
    DATA = np.delete(DATA, CHANNELS_TO_REMOVE, axis=-1)  # Remove in last axis (channels dimension)
    print('Completed. Dimensions of data after removing channels: ', DATA.shape)

    # Normalize the complete dataset at once
    MINIMUM_OF_DATA = np.min(DATA)
    MAXIMUM_OF_DATA = np.max(DATA)

    print('Minimum of the data before normalization: ', MINIMUM_OF_DATA)
    print('Maximum of the data before normalization: ', MAXIMUM_OF_DATA)
    DATA = (DATA - MINIMUM_OF_DATA) / (MAXIMUM_OF_DATA - MINIMUM_OF_DATA)  # Min-Max normalization
    print('Normalized!')

    MINIMUM_OF_DATA = np.min(DATA)
    MAXIMUM_OF_DATA = np.max(DATA)
    print('Minimum of the data after normalization: ', MINIMUM_OF_DATA)
    print('Maximum of the data after normalization: ', MAXIMUM_OF_DATA)

    ### DATA ARRANGMENT ###

    if DATA_PROCESSING_MODE == '1D_PROCESSING':
        print('Before dimension arangement: ', DATA.shape)
        DATA_FLATTENED = DATA.reshape(DATA.shape[0] * DATA.shape[1] * DATA.shape[2],
                                      DATA.shape[3])  # Dimensions: PIXELS x CHANNELS
        print('After dimension arangement: ', DATA_FLATTENED.shape)
        # DATA_FLATTENED will be passed to the model

    elif DATA_PROCESSING_MODE == '3D_PROCESSING':
        # Necessary to patch the images first, as follows:
        #   1) Padd the images' borders
        #   2) Patch the images

        print('DATA before padding has dimensions: ', DATA.shape)

        print('Padding images...')
        # VERY IMPORTANT! Do not modify the call to the next method for padding the data unless aware of the effects
        # of introducing changes. The call is already fixed to work under the 3D-Processing scenario.
        PATCH_SIZE = 48
        DATA_PADDED, DUMMY_LABELS_PADDED, \
            NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED, NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED = \
            utils_o.cropping_or_padding_of_spatial_resolution(
                DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE='3D-PROCESSING', \
                ENABLE_SHOULD_PAD=True, \
                DATA=DATA, \
                LABELS=-1 * np.ones((DATA.shape[0], DATA.shape[1], DATA.shape[2])), \
                PATCH_SIZE=PATCH_SIZE, \
                PADDING_TECHNIQUE='CONSTANT_PADDING_EXTENDING_EDGES')
        # Note: LABELS are just some dummy labels as the deployment set does not have any ground-truth labels.
        # Dimensions for DATA: IMAGES x LINES x SAMPLES x CHANNELS
        # Dimensions for LABELS: IMAGES x LINES x SAMPLES (categorical)
        # Resulting outputs from the method:
        #   DATA has the same number of images and channels, yet the lines and samples have been padded
        #   DUMMY_LABELS: not relevant for this notebook

        print('Padding completed!...')

        LINES_AFTER_PADDING = DATA_PADDED.shape[1]
        SAMPLES_AFTER_PADDING = DATA_PADDED.shape[2]

        print('Number of lines after padding: ', LINES_AFTER_PADDING)
        print('Number of samples after padding: ', SAMPLES_AFTER_PADDING)
        print('DATA after padding has dimensions: ', DATA_PADDED.shape)

        # At this point, the following parameters are important for patching:
        #   PATCH_SIZE, NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED, NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED

        # Before patching, we first need to ensure that PATCH_SIZE, NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED, and
        # NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED are integer in order to create the dimensions of the arrays that will contain
        # the patched data.
        PATCH_SIZE = int(PATCH_SIZE)
        NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED = int(NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED)
        NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED = int(NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED)

        # We now patch the data next
        print('Patching the data...')

        DATA_PATCHED, DUMMY_LABELS_PATCHED = \
            utils_o.patch_data_and_annotations(
                DATA=DATA_PADDED, \
                ANNOTATIONS=DUMMY_LABELS_PADDED, \
                PATCH_SIZE=PATCH_SIZE, \
                NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED=NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED, \
                NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED=NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED
            )

        # DATA_PADDED is in 4D: IMAGES x LINES x SAMPLES x CHANNELS - yet LINES and SAMPLES have been padded
        # DUMMY_LABELS_PADDED: not relevant
        # The resulting output would be:
        # DATA_PATCHED has dimensions:
        #   IMAGES x NUMBER_OF_PATCHES_IN_LINES_DIRECTION x NUMBER_OF_PATCHES_IN_SAMPLES_DIRECTION x
        #           x PATCH_SIZE x PATCH_SIZE x CHANNELS
        # DUMMY_LABELS_PATCHED: not relevant
        print('Patching completed!')

        # Now flatten the patches as the models (2D-CNNs) will take input: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNELS

        print('Data before flattening patches: ', DATA_PATCHED.shape, ', and dtype: ', DATA_PATCHED.dtype)
        DATA_FLATTENED = \
            DATA_PATCHED.reshape(DATA_PATCHED.shape[0] * DATA_PATCHED.shape[1] * DATA_PATCHED.shape[2], \
                                 PATCH_SIZE, PATCH_SIZE, DATA_PATCHED.shape[5])
        # DATA_PATCHED has dimensions:
        #   IMAGES x NUMBER_OF_PATCHES_IN_LINES_DIRECTION x NUMBER_OF_PATCHES_IN_SAMPLES_DIRECTION x
        #           x PATCH_SIZE x PATCH_SIZE x CHANNELS
        # The resulting output of the method is DATA_FLATTENED with dimensions:
        #    NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNELS
        print('Data after flattening patches: ', DATA_FLATTENED.shape, ', and dtype: ', DATA_FLATTENED.dtype)

        # Note: no need to flatten the patches for the labels as they are mere dummy annotations used previously to be able to call the methods for
        #       padding and patching (these methods expect to receive both data and labels at the same time).

    ### INFERENCE ###

    print('Tip: make sure GPU is enabled next for faster inference.\n')
    print('Accessible devices by Tensorflow (check if any GPU is in the list):\n ',
              tf.config.list_physical_devices())
    print(
            '\nSegmentation inference running (it may take some time depending on the model, GPU, and number of images to segment)...')

    print('Number of images to segment: ', NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, \
              '\nFor instance, if each image takes for instance 15s (as in 1D-Justo-LiuNet), then the segmentation of e.g. 30 images would take over 7 minutes.')
    print('\nReminder of results: for 3D-Processing inferece time is significantly smaller')
    STARTING_TIME = time.time()
    PREDICTION = model_classifier.predict(DATA_FLATTENED)
    FINISH_TIME = time.time()

    print('Inference completed!')
    INFERENCE_TIME = FINISH_TIME - STARTING_TIME
    print(f'Inference time to segment the image: {round(INFERENCE_TIME, 2)} seconds.')

    ### POST: CATEGORICAL PREDICTIONS AND CONFIDENCE ###

    print('Predictions from inference have dimensions: ', PREDICTION.shape, ', and dtype: ', PREDICTION.dtype)
    PREDICTED_PROBABILITY = np.max(PREDICTION,
                axis=-1)  # Largest probability, referred henceforth as 'confidence probability', which

    # determines the degree of confidence of the model when making a prediction for a pixel
    PREDICTED_CATEGORICAL_CLASS = np.uint8(np.argmax(PREDICTION, axis=-1))  # The final categorical prediction

    print('Dimensions when computing ONLY the categorical prediction for each pixel: ',
    PREDICTED_CATEGORICAL_CLASS.shape, ', and dtype: ', PREDICTED_CATEGORICAL_CLASS.dtype)
    print('Dimensions for the confidence probabilities: ', PREDICTED_PROBABILITY.shape, ', and dtype: ',
    PREDICTED_PROBABILITY.dtype)

    ### POST: DIMENSION ARRANGMENT ###
    # At this point, arange dimensions for PREDICTED_CATEGORICAL_CLASS and PREDICTED_PROBABILITY
    if DATA_PROCESSING_MODE == '1D_PROCESSING':
        print('Unflattening data points...')
        PREDICTED_CATEGORICAL_CLASS = PREDICTED_CATEGORICAL_CLASS.reshape(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, LINES,
        SAMPLES)
        PREDICTED_PROBABILITY = PREDICTED_PROBABILITY.reshape(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, LINES, SAMPLES)

        print('UNFLATTENED OK - Dimensions for segmented dataset (categorical): ',
        PREDICTED_CATEGORICAL_CLASS.shape, ', and dtype: ', PREDICTED_CATEGORICAL_CLASS.dtype)
        print('UNFLATTENED OK - Dimensions for the confidence probabilities for the segmented dataset: ',
        PREDICTED_PROBABILITY.shape, ', and dtype: ', PREDICTED_PROBABILITY.dtype)

    elif DATA_PROCESSING_MODE == '3D_PROCESSING':

        # Stage 1: Unflatten the patches
        print('Unflattening patches...')
        PREDICTED_CATEGORICAL_CLASS = \
        PREDICTED_CATEGORICAL_CLASS.reshape(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, \
        NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED,
        NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED, \
        PATCH_SIZE, PATCH_SIZE)
        PREDICTED_PROBABILITY = \
        PREDICTED_PROBABILITY.reshape(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, \
        NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED,
        NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED, \
        PATCH_SIZE, PATCH_SIZE)
        # We ensure to have the following dimensions:
        #   NUMBER_OF_IMAGES x NUMBER_OF_PATCHES_IN_LINES_DIMENSIONS x NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSIONS x PATCH_SIZE x PATCH_SIZE

        print('UNFLATTENED patches (categorical) have now dimension: ', PREDICTED_CATEGORICAL_CLASS.shape,
        ', and dtype: ', PREDICTED_CATEGORICAL_CLASS.dtype)
        print('UNFLATTENED patches (confidence probability) have now dimension: ', PREDICTED_PROBABILITY.shape,
        ', and dtype: ', PREDICTED_PROBABILITY.dtype)

        # Stage 2: Unpatch
        print('\nThe unpatching next is for the categorical predictions:')
        PREDICTED_CATEGORICAL_CLASS = \
        utils_o.unpatch_predictions(PREDICTIONS_patched=PREDICTED_CATEGORICAL_CLASS, \
        TARGET_shape=(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, LINES_AFTER_PADDING,
        SAMPLES_AFTER_PADDING), \
        PATCH_SIZE=PATCH_SIZE)
        print('\nThe unpatching next is for the confidence probabilities:')
        PREDICTED_PROBABILITY = \
        utils_o.unpatch_predictions(PREDICTIONS_patched=PREDICTED_PROBABILITY, \
        TARGET_shape=(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET, LINES_AFTER_PADDING,
        SAMPLES_AFTER_PADDING), \
        PATCH_SIZE=PATCH_SIZE)

    ### VISUALIZE ###

    OUTPUT_PATH = "./images"
    for iterator_segmented_image in range(NUMBER_OF_IMAGES_IN_DEPLOYMENT_SET):
        print(PATHS_TO_DATA[iterator_segmented_image])
        # Visualise next the results from the segmentation model
        VISUALISE_SEGMENTATION_RESULTS(IMAGE=iterator_segmented_image, \
        FONTSIZE=14, \
        FIGURE_TITLE='Segmentation results', \
        FIGSIZE=(17, 6), \
        CHANNEL=89, \
        DATA=DATA, \
        SEGMENTED_IMAGE=PREDICTED_CATEGORICAL_CLASS[iterator_segmented_image, :,
        :], \
        CONFIDENCE_PROBABILITY_SPATIAL_DOMAIN=PREDICTED_PROBABILITY[
        iterator_segmented_image, :, :], \
        LINES=PREDICTED_CATEGORICAL_CLASS.shape[1], \
        SAMPLES=PREDICTED_CATEGORICAL_CLASS.shape[2],
        OUTPUT_PATH=OUTPUT_PATH)



if __name__ == "__main__":
    main()