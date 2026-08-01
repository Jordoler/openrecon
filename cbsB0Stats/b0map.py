import ismrmrd
import os
import logging
import traceback
import numpy as np
import base64
import mrdhelper
import constants
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
import subprocess

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, mrdHeader):
    logging.info("Config: \n%s", config)
    sendOriginals = mrdhelper.get_json_config_param(config, 'SendOriginals', default=True, type='bool')
    slice_max = mrdHeader.encoding[0].encodingLimits.slice.maximum
    te1_array = np.zeros((slice_max),dtype=object)
    te2_array = np.zeros((slice_max),dtype=object)
    ph_array = np.zeros((slice_max),dtype=object)
    TEs = []
    b0mapSeries = 0
    if (mrdHeader.sequenceParameters.TE is not None):
        TEs = mrdHeader.sequenceParameters.TE
        logging.info(f"TE List collected from sequence header: {TEs}")
        if len(TEs)>2:
            logging.info(f"Too many TE's keep first two {TEs[0:2]}")
        TEs = [float(min(TEs)), float(max(TEs))]
    else:
        TEs = mrdhelper.get_json_config_param(config, 'TE1', default=10.0, type='float'), mrdhelper.get_json_config_param(config, 'TE2', default=12.0, type='float') 
        logging.info(f"Could not find TE List, Using User defined parameters TE1 = {TEs[0]}, TE2 = {TEs[1]}")

    # mrdHeader should be xml formatted MRD header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("MRD header: \n%s", mrdHeader.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(mrdHeader.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            mrdHeader.encoding[0].trajectory, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.x, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.y, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.z, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.x, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.y, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z)
    except:
        logging.info("Improperly formatted MRD header: \n%s", mrdHeader)
    
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                raise Exception("Raw k-space data is not supported by this application, switch reconstruction emitter to 'image'")

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                    
                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if ((item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0)):
                    meta = ismrmrd.Meta.deserialize(item.attribute_string)
                    iceHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
                    echo_number = int(mrdhelper.extract_minihead_long_param(iceHeader, 'EchoNumber'))
                    slice_no = item.slice
                    if item.image_series_index > b0mapSeries:
                        b0mapSeries = item.image_series_index
                    if echo_number == 1:
                        if (type(te1_array[slice_no]) is not int):
                            item_in = te1_array[slice_no]
                            item_in_meta = ismrmrd.Meta.deserialize(item_in.attribute_string)
                            item_in_iceHeader = base64.b64decode(item_in_meta['IceMiniHead']).decode('utf-8')
                            item_in_isCorrected =  mrdhelper.extract_minihead_string_param(item_in_iceHeader,"SequenceDescription").endswith("_ND")
                            item_isCorrected = mrdhelper.extract_minihead_string_param(iceHeader,"SequenceDescription").endswith("_ND")
                            if not item_in_isCorrected and item_isCorrected:
                                # replace already collected te1_item if new te1_item is corrected
                                te1_array[slice_no] = item 
                        else:
                            te1_array[slice_no] = item              

                if (item.image_type is ismrmrd.IMTYPE_PHASE):
                    meta = ismrmrd.Meta.deserialize(item.attribute_string)
                    iceHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
                    echo_number = int(mrdhelper.extract_minihead_long_param(iceHeader, 'EchoNumber'))
                    if echo_number == 2:
                        slice_no = item.slice
                        ph_array[slice_no] = item

                if sendOriginals:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()
                    connection.send_image(item)

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                raise Exception("Waveform data is not supported by this application, switch reconstruction emitter to 'image'")

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        if type(ph_array[-1] is not int) and type(te1_array[-1] is not int):
            logging.info("Calculating B0map")
            image = process_b0map(te1_array,ph_array,TEs,b0mapSeries,config,mrdHeader)
            connection.send_image(image)
        else:
            logging.error("Insufficent images collected, B0map was not calculated")
        connection.send_close()

def process_b0map(te1_array,ph_array,TEs,b0mapSeries,config,mrdHeader):
    logging.info(f'----------------------------------------------------------------------------------------------')
    logging.info(f'     process_b0map called with 2x mag and 1x phase diff each with {len(te1_array)} images')
    logging.info(f'----------------------------------------------------------------------------------------------')

    Verbose = mrdhelper.get_json_config_param(config, 'VerboseLogging', default=False, type='bool')
    debug = mrdhelper.get_json_config_param(config, 'debug', default=False, type='bool')
    Num_ero = mrdhelper.get_json_config_param(config, 'NumEro', default=3, type='int')
    Num_Dil = mrdhelper.get_json_config_param(config, 'NumDil', default=2, type='int')
    FracIntens = float(mrdhelper.get_json_config_param(config, 'FracIntens', default=0.4, type='float'))
    gaussSmoothing = mrdhelper.get_json_config_param(config, 'gaussianSmoothing', default=False, type='bool')

    delTE = TEs[1]-TEs[0]
    xform = np.eye(4)

    meta_0 = ismrmrd.Meta.deserialize(te1_array[0].attribute_string)
    iceHeader_0 = base64.b64decode(meta_0['IceMiniHead']).decode('utf-8')
    RescaleIntercept = float(mrdHeader.extract_minihead_double_param(iceHeader_0,'RescaleIntercept'))
    RescaleSlope = float(mrdHeader.extract_minihead_double_param(iceHeader_0,'RescaleSlope'))
    BitsStored = 12
    BitsStored = int(mrdHeader.extract_minihead_long_param(iceHeader_0,'BitsStored'))

    logging.info(f"B0MAP_LOG: Params Collected:\n b0mapSeries: {b0mapSeries}\n delTE: {delTE}\n Num_ero: {Num_ero}\n Num_Dil: {Num_Dil} \nRescale Intercept: {RescaleIntercept} \nRescale Slope: {RescaleSlope}\nBitsStored: {BitsStored}")

    # Create folder, if necessary
    if not os.path.exists(debugFolder) and debug:
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Note: The MRD Image class stores data as [cha z y x]
    # Extract image data into a 5D array of size [img cha z y x]
    
    T1data = np.stack([item.data                              for item in te1_array])
    T1head = [item.getHead()                                  for item in te1_array]
    T1meta = [ismrmrd.Meta.deserialize(item.attribute_string) for item in te1_array]
    PHdata = np.stack([item.data                              for item in ph_array])

    del ph_array
    del te1_array

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    #data = data.transpose((3, 4, 2, 1, 0))

    #Transpose data arrays
    T1data, PHdata = [data.transpose((3, 4, 2, 1, 0))[:, :, 0, 0, :] for data in [T1data, PHdata]]
    logging.info(f"B0map: Data arrays transposed")
    if debug:
        np.save(debugFolder + "/" + "imgT1Orig.npy", T1data)
        np.save(debugFolder + "/" + "imgPHOrig.npy", PHdata)
    if Verbose:
        cen_e = tuple(s//2 for s in T1data.shape) # center element
        logging.debug("Original T1 image data is size %s" % (T1data.shape,))
        logging.debug("Original Phase image data is size %s" % (PHdata.shape,))
        logging.info("After slice and transpose: %s", T1data.shape)
        logging.info(f"B0map: center element; PHdata{cen_e}: {PHdata[cen_e]}")
        logging.info(f"B0map: center element; T1data{cen_e}: {T1data[cen_e]}")

    PHdata = PHdata.astype(np.float64)
    T1data = T1data.astype(np.float64)

    #Make Frequency Map
    PHdata *= RescaleSlope
    PHdata += RescaleIntercept
    PHscale = 1.0 / (2.0 * np.abs(RescaleIntercept) * delTE * 1e-3) #not a real ph-scale!
    PHdata *= PHscale
    logging.info(f"B0map: Created frequency map")
    if Verbose:
        logging.info(f"B0map: center frequency element; freq_data{cen_e}: {PHdata[cen_e]}")
    if debug:
            np.save(debugFolder + "/" + "imgFreq.npy", PHdata)

    #Make Brain Mask with bet2 and scipy.ndimage
    T1data = T1data.transpose(1,0,2) # to [x,y,z]
    try:
        nib.save(nib.nifti1.Nifti1Image(T1data,xform),'temp_t1.nii')
        if os.path.isfile('temp_t1.nii'):
            logging.info("B0map: saved mag1 image to nifti")
    except Exception as e:
        logging.error("B0map: Could not find mag1. to nifti\n{e}")
    try: 
        subprocess.run(["bet2","temp_t1.nii","temp_bm","-m","-n","-f",f"{round(FracIntens,1)}"],check=True)
        if os.path.isfile('-n.nii.gz'):
            logging.info("b0map: performed bet2 brain mask on mag1 data")
    except Exception as e:
        logging.error(f"b0map: Failed to run bet2\n{e}")
    
    BMask = None
    try:
        BMask_load = nib.load("-n.nii.gz")
        BMask = BMask_load.get_fdata()
        if Verbose:
            logging.info(f"b0map: center element of Raw Bmask; Bmask{cen_e}: {BMask[cen_e]}")
    except Exception as e:
            logging.info(f"b0map: Failed to import brain mask nifti\n{e}")
    if BMask is not None:
        BMask = binary_erosion(BMask, iterations=Num_ero)
        BMask = binary_dilation(BMask,iterations=Num_Dil)
    BMask = BMask.transpose((1,0,2)).astype(np.float64)
    if debug:
        np.save(debugFolder + "/" + "imgBMask.npy", BMask)
    if Verbose:
        logging.info(f"B0map: center element of final Brain mask; BMask{cen_e}: {BMask[cen_e]}")
    logging.info(f"B0map: Constructed Brain Mask with {Num_ero} erosions and {Num_Dil} dilations")

    #Make B0map
    if Verbose:
        logging.info("B0map: Brain mask shape - %s", BMask.shape)
        logging.info("B0map: Freq map shape - %s", PHdata.shape)
    if BMask is None:
        logging.info("B0map: Failed to calculate BrainMask resorting to B0map without a brain mask...")
        B0map = (PHdata).astype(np.float64)
    else:
        B0map = ((BMask*PHdata)).astype(np.float64)
        logging.info("B0map: Constructed B0map")
    if Verbose:
        logging.info(f"B0map: center b0map element; B0map{cen_e}: {B0map[cen_e]}")
    if debug:
        np.save(debugFolder + "/" + "imgb0map.npy", B0map)

    if gaussSmoothing:
        # y x z)
        for z in range(B0map.shape[-1]):
            B0map[:,:,z] = gaussian_filter(B0map[:,:,z], sigma=1.0)
        logging.info("B0map: performed guassian smoothing")

    B0map = B0map[:,:,None,None,:].astype(np.float32)

    B0map = np.clip(B0map,0,4096)

    #b0map mean and std
    mask = BMask.astype(bool) if BMask is not None else np.zeros_like(PHdata,dtype=bool)
    masked_freq = PHdata[mask]
    B0min,B0max,B0mean,B0std = None,None,None,None
    if masked_freq.size >0:
        B0min = np.min(masked_freq)
        B0max = np.max(masked_freq)
        B0mean = np.mean(masked_freq)
        B0std = np.std(masked_freq)
        logging.info(f"B0map: Min: {B0min}, Max: {B0max}")
        logging.info(f"B0map: Mean: {B0mean:.5f}, STD: {B0std:.5f}")
    else:
        logging.error(f"B0map: Failed to calculate stats for B0map")
    logging.info("B0map: calculated B0map stats")

    # Prepare b0map for output
    InitialItem = 0
    B0mapOut = [None] * B0map.shape[-1]
    for item in range(B0map.shape[-1]):
        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        tmp2 = B0map[...,item].transpose((3,2,0,1))[0,0,:,:] # Crops from [z,char,y,x,(img)] to [y,x,(img)] -> ..,item indicate this the (img^{th}) item
        B0mapOut[item] = ismrmrd.Image.from_array(tmp2,transpose = False)

        if item == InitialItem and Verbose:
            try:
                logging.info("B0map: Assessing ismrmrd mapping to np.astype...")
                logging.info(f"B0map: B0mapOut[0].data_type is {B0mapOut[item].data_type}")
                logging.info(f"B0map: B0map[...,item].dtype is {B0map[...,item].dtype}")
                logging.info(f"B0map: logging tmpMeta: {T1meta[item]}")
                logging.info(f"B0map: logging B0Header: {T1head[item]}")
            except Exception as e:
                logging.info(f"B0map: Failed to assess ismrmrd mapping to np.astype:\n{e}")

        B0Header = T1head[item]
        B0Header.data_type = B0mapOut[item].data_type
        B0Header.image_series_index = b0mapSeries
        tmpMeta = T1meta[item]
        
        B0mapOut[item].setHead(B0Header)

        # Create a copy of the original ISMRMRD Meta attributes from TE1 and update where possible
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'B0MAP']
        tmpMeta['WindowCenter']                   = str(np.round(((B0max - B0min)/2)))
        tmpMeta['WindowWidth']                    = str((B0max)+1)
        tmpMeta['SequenceDescriptionAdditional']  = 'OPENRECON_B0MAP'
        tmpMeta['Keep_image_geometry']            = 1
        tmpMeta['ImageComments']                  = f"Mean: {B0mean:.3f}, STD: {B0std:.3f}" # Change the image comments to mean and SD
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(B0Header.read_dir[0]), "{:.18f}".format(B0Header.read_dir[1]), "{:.18f}".format(B0Header.read_dir[2])]
                    
        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(B0Header.phase_dir[0]), "{:.18f}".format(B0Header.phase_dir[1]), "{:.18f}".format(B0Header.phase_dir[2])]
                
        metaXml = tmpMeta.serialize()
                    
        B0mapOut[item].attribute_string = metaXml
    logging.info(f"B0map: Returning B0map")
    return B0mapOut