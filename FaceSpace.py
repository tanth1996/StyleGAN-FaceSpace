import av
from os import listdir
from os.path import isfile, join
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

from crop_align_face import CropAlignFace
from train_latent_space import FaceLatents, TrainLatentsEnc
from models.stylegan import StyleGAN
from plot_utils import ShowModelOutput
from time import time

def GetFramesFromVideo(VideoPath, FramesDir, StartFrame=0, StopFrame=None, FrameStep=10, Rotate=0):
    """Extracts frames from a given video path, VideoPath, to a folder path, FramesDir. 
    
    Optional Arguments:
    StartFrame -- Index of video frame to begin extraction from
    StopFrame -- Index of video frame where extraction stops
    FrameStep -- Number of frames to step over from one extracted frame to the next
    Rotate -- Anticlockwise rotation of frame in degrees during extraction
    """

    # Get video contrainer (generator object)
    VideoContainer = av.open(VideoPath)
    for Frame in VideoContainer.decode(video=0):
        # Do nothing until StartFrame is reached
        if Frame.index == StartFrame:
            # Save a frame as image every FrameStep
            for Frame in VideoContainer.decode(video=0):
                if (Frame.index - StartFrame) % FrameStep == 0:
                    Img = Frame.to_image().rotate(Rotate, expand=True)
                    Img.save(FramesDir + '/frame-%04d.jpg' % Frame.index)
                # Break if StopFrame is specified and is reached
                if StopFrame is not None and Frame.index >= StopFrame:
                    break
            break
                

def PreprocessFaceDir(FaceDir, AlignedDir):
    """Crops and aligns images of faces based on facial landamrks given a containing directory, FaceDir.

    Outputs aligned face images to a new directory, AlignedDir.
    Aligned images are 1024x1024 px in png format.
    This preprocessing is similar to the ffhq dataset preprocessing procedure.
    """
    CropAlignFace(FaceDir, AlignedDir, LandmarksModelPath='landmarks/shape_predictor_68_face_landmarks.dat')


def MakeFaceData(AlignedFaceDir=None, GetFacesFeatures=False, Zdims=None, GetLandmarks=False, W_meanPath='W_mean.pt',
                 LoadPath=None, PrintInterval=1, UseCuda=True):
    """Returns an instance of FaceLatents containing data required for training.

    Either AlignedFaceDir or LoadPath must be specified at a minimum for this function.

    Optional Arguments:
    AlignedFaceDir -- Folder containing cropped and aligned faces; size of dataset is inferred from number of images in this folder
    GetFacesFeatures -- Boolean flag specifying if feature maps are to be stored and extracted
    Zdims -- Dimensionality of input Zspace, eg. number of landmarks as inputs, number of basis faces etc.
    GetLandmarks -- Boolean flag specifying if facial landmarks should be extracted from faces
    W_meanPath -- Path for a .pt file containing a 512 dimensional tensor representing the mean latent W vector to initialise the Wspace as
    LoadPath -- Path of pickled FaceLatents object to load from; all other arguments are ignored if specified.
    PrintInterval -- Interval between faces being printed during initialisation.
    UseCuda -- Boolean flag specifying if FaceLatents object should be sent to GPU.
    """
    assert(not(AlignedFaceDir is None and LoadPath is None)), 'Both AlignedFaceDir and LoadPath arguments unspecified, at least one of them must be.'
        
    if LoadPath is not None:
        # Load data from path
        print('Load path specified; ignoring all other input arguments and loading face data from file...')
        with open(LoadPath, 'rb') as FaceDataPath:
            FaceData = pickle.load(FaceDataPath)
        return FaceData
    
    # Make face data object based on given folder
    print('Getting tensors of (aligned) face images')
    TargetPaths = [join(AlignedFaceDir, f) for f in listdir(AlignedFaceDir) if isfile(join(AlignedFaceDir, f))]
    TargetPaths = sorted(TargetPaths)
    Size = len(TargetPaths)
    if Zdims is None:
        print('ZDims unspecified, will be inferred from number of images provided.')
        print('Zdims set equal to Size: ', Size)
        Zdims = Size

    FaceData = FaceLatents(Zdims=Zdims, Size=Size, UseCuda=UseCuda)

    if GetLandmarks:
        FaceData.GetFacesFromPaths(TargetPaths, PrintFaces=False)
        print('Getting face landmarks...')
        FaceData.GetLandmarks(PrintInterval=PrintInterval)
    else:
        FaceData.GetFacesFromPaths(TargetPaths, PrintInterval=PrintInterval)

    if GetFacesFeatures:
        print('Note: Feature Maps take up a large amount of memory, and will result in a large object file if saved.')
        if AlignedFaceDir is None:
            print('Warning: No face image directory specified, features cannot be extracted')
        else:
            print('Getting feature maps for training...')
            FaceData.GetFeatureMaps()
            print('Done')

    # Set Wspace to the mean latent W vector
    if W_meanPath is not None:
        print('Setting Wspaces to be the mean latent vector from ', W_meanPath)
        W_mean = torch.load(W_meanPath).view((1,1,512))
        Wspace = W_mean.repeat((FaceData.Size,18,1))
        FaceData.Wspace = Wspace
        FaceData.Wspace.requires_grad = True

    return FaceData


def MakeFaceModel(FaceData=None, Zdims=None, WeightLoadPath=None, Landmarks=False, UseCuda=True, \
                  PretrainedWeightPath='weights/karras2019stylegan-ffhq-1024x1024.for_g_all.pt'):
    """Returns a StyleGAN with an encoding network based on dimensions of a provided FaceData object or explicitly stated dimensions, Zdims.
    
    Either FaceData or Zdims must be specified at a minimum for this function.

    Optional Arguments:
    FaceData -- FaceLatents object; FaceModel dimensions will be inferred from attributes of this object
    Zdims -- Dimensionality of input Zspace
    WeightLoadPath -- Path containing model weights to load
    Landmarks -- Boolean flag specifying if FaceModel should be constructed with a landmarks encoder
    UseCuda -- Boolean flag specifying if FaceModel should be sent to GPU
    PretrainedWeightPath -- Path containing pretrained StyleGAN model weights
    """
    assert(not(FaceData is None and Zdims is None)), 'Both FaceData and CompDims arguments unspecified, at least one of them must be.'
    if UseCuda:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            raise Warning('UseCuda was set true but cuda is not available, using cpu')
    else:
        device = 'cpu'

    if Zdims is None and not Landmarks:
        Zdims = FaceData.Zdims
        print('No Zdims explicitly specified, inferring Zdims from FaceData, Zdims=%d' %Zdims)
    elif Zdims is None and Landmarks:
        Zdims = 136
        print('No Zdims explicitly specified, assuming 68 landmarks, Zdims=%d' %Zdims)

    # Make StyleGAN with (dimension decompression) encoder
    if not Landmarks:
        FaceModel = StyleGAN(decompress=True, dims=Zdims, weight_path=PretrainedWeightPath).to(device)
    # Make StyleGAN with facial landmarks encoder for pose conditioning
    else:
        FaceModel = StyleGAN(landmarks=True, weight_path=PretrainedWeightPath).to(device)
    
    if WeightLoadPath is not None:
        # Load weights from path
        print('Loading model weights from path...')
        FaceModel.load_state_dict(torch.load(WeightLoadPath))
        print('Done')
    
    return FaceModel


def TrainFaceModel(FaceModel, FaceData, BatchSize=2, Epochs_n=200, Cycles=2, Landmarks=False,
                   LearnRate=1e-2, LearnRateAnnealFactor=0.1, TargetPropAtStartOnly=True, PixelLossIncrease=False, 
                   TrainZ=False, PrintInterval=25, MaxBatchPrint=10, UseCuda=True):
    """"Calls the main training function for a specified number of epochs and cycles with annealed learning rates.

    Takes a StyleGAN model with encoder, FaceModel and FaceLatents object, FaceData as inputs.
    With default optional arguments, uses experimentally tuned weights with an additional fine tuning cycle based on higher pixel loss

    Optional Arguments:
    BatchSize -- Batch size during training
    Epochs_n -- Number of epochs per cycle
    Cycles -- Number of cycles with successively annealed learning rates
    LearnRate -- Initial learning rate for Adam optimiser
    LearnRateAnnealFactor -- Factor that learning rate is multiplied by for successive cycles
    TargetPropAtStartOnly -- Boolean flag specifying if target propagation loss will be used only for an initial phase
    PixelLossIncrease -- Boolean flag specifying if additional fine tuning cycle with increased pixel loss is performed
    PrintInterval -- Interval between epochs whereby loss metrics and output images are printed out
    TrainZ -- Boolean flag specifying if the Zspace should be trained
    MaxBatchPrint -- Maximum number of batches to print during an epoch where printing occurs
    UseCuda -- Boolean flag specifying if training should use CUDA where possible
    """
    TimeStart = time()
    StoredTargetFeatures = False if FaceData.FeatureMaps == [] else True
    if not StoredTargetFeatures:
        print('Note: No FeatureMaps stored in FaceData, will extract target feature maps each batch')

    if Landmarks:
        print('Training for landmark encoding (pose conditioning)')
        Weight_feature = 1e2
        Weight_pixel = 1e2
        Weight_tp = 2e4
    else:
        print('Training for basis face encoding (synthetic face generation)')
        Weight_feature = 1e2
        Weight_pixel = 1e2
        Weight_tp = 1e4

    # Train with target propagation loss only at the start
    if TargetPropAtStartOnly:
        print('Training with target propagation loss only for this initial phase.')
        TrainLatentsEnc(FaceModel, FaceData, Epochs_n, LearnRate, LearnRateW=LearnRate, BatchSize=BatchSize,\
                        Weight_feature=Weight_feature, Weight_pixel=Weight_pixel, Weight_tp=Weight_tp,\
                        TrainZ=TrainZ, StoredTargetFeatures=StoredTargetFeatures,\
                        PrintInterval=PrintInterval, MaxBatchPrint=MaxBatchPrint, UseCuda=UseCuda)
        Weight_tp = 0

    # Initial training with lower pixel loss
    for Cycles_i in range(1, Cycles+1):
        print('Cycle: ', Cycles_i, 'LR: ', LearnRate)
        TrainLatentsEnc(FaceModel, FaceData, Epochs_n, LearnRate, LearnRateW=LearnRate, BatchSize=BatchSize,\
                        Weight_feature=Weight_feature, Weight_pixel=Weight_pixel, Weight_tp=Weight_tp,\
                        TrainZ=TrainZ, StoredTargetFeatures=StoredTargetFeatures,\
                        PrintInterval=PrintInterval, MaxBatchPrint=MaxBatchPrint, UseCuda=UseCuda)
        LearnRate *= LearnRateAnnealFactor

    # Fine tune with higher pixel loss
    if PixelLossIncrease:
        if Weight_pixel > 0:
            Weight_pixel *= 100
        if Weight_pixel < 1e2:
            Weight_pixel = 1e2 # Use a minimum weight of 1e2
        print('Fine tuning with higher pixel loss.')
        TrainLatentsEnc(FaceModel, FaceData, Epochs_n // 2, LearnRate, LearnRateW=LearnRate, BatchSize=BatchSize,\
                        Weight_feature=Weight_feature, Weight_pixel=Weight_pixel, Weight_tp=Weight_tp,\
                        TrainZ=TrainZ, StoredTargetFeatures=StoredTargetFeatures,\
                        PrintInterval=PrintInterval, MaxBatchPrint=MaxBatchPrint, UseCuda=UseCuda)
    
    print('Total time elapsed for all training cycles: ', time() - TimeStart, ' s')


def SaveFaceSpace(FaceData=None, DataPath=None, FaceModel=None, ModelWeightsPath=None, SaveFeatureMaps=False):
    """Saves a face data object and/or model weights to specified paths.
    
    Either FaceData and its corresponding DataPath, or FaceModel and its corresponding ModelWeightsPath must be specifed at a minimum for this function.
    
    Optional arguments:
    FaceData -- FaceLatents object to pickle and save
    DataPath -- Path where FaceData is to be pickled and saved to
    FaceModel -- Network whose weights are to be saved
    ModelWeightsPath -- Path where FaceModel's weights are to be saved to
    SaveFeatureMaps -- Boolean flag specifying if feature maps in FaceData should be saved, note: feature maps take up a lot of storage space
    """
    assert(not(FaceData is None and FaceModel is None)), "At least one argument among FaceData and FaceModel must be specified"

    assert(not(FaceData is not None and DataPath is None)), "Save path for FaceData must be specified"
    assert(not(FaceModel is not None and ModelWeightsPath is None)), "Save path for FaceModel must be specified"

    if FaceData is not None:
        if not SaveFeatureMaps:
            FaceData.FeatureMaps = []
        with open(DataPath, 'wb') as SavePath:
            pickle.dump(FaceData, SavePath)
        print('FaceData saved to ', DataPath)

    if FaceModel is not None:
        torch.save(FaceModel.state_dict(), ModelWeightsPath)
        print('FaceModel saved to ', ModelWeightsPath)


def LoadFaceSpace(DataPath=None, ModelWeightsPath=None, Zdims=None, Landmarks=False, UseCuda=True, GetFeatureMaps=False):
    """Loads and returns a face data object and optionally a model with corresponding weights from the specified paths.

    Either DataPath or ModelWeightsPath must be specified at a minimum for this function

    Optional arguments:
    DataPath -- Path of saved pickled FaceLatents object to be loaded
    ModelWeightsPath -- Path of model weights to be loaded
    Landmarks -- Boolean flag specifying if StyleGAN model to be loaded should have a landmarks encoder
    UseCuda -- Boolean flag specifying if loaded model should be sent to GPU
    GetFeatureMaps -- Boolean flag specifying if feature maps should be extracted for FaceLatents object after loading it
    """
    assert(not(DataPath is None and ModelWeightsPath is None)), "At least one argument among DataPath and ModelWeightsPath must be specified"
    assert(not(ModelWeightsPath is not None and DataPath is None and Zdims is None)), "Either a DataPath to load FaceData from or Zdims must be specified to load a model"
    
    if DataPath is not None:
        with open(DataPath, 'rb') as LoadPath:
            FaceData = pickle.load(LoadPath)
        if GetFeatureMaps:
            print('Getting feature maps for training...')
            FaceData.GetFeatureMaps()
            print('Done')

    else:
        FaceData = None
    
    if ModelWeightsPath is not None:
        try:
            FaceModel = MakeFaceModel(FaceData=FaceData, Zdims=Zdims, WeightLoadPath=ModelWeightsPath, Landmarks=Landmarks, UseCuda=UseCuda)
        except:
            # Try again this time filter out unnecessary keys
            print('Model weight mismatch, trying again after filtering out extra keys')
            FaceModel = MakeFaceModel(FaceData)
            WeightsDict = torch.load(ModelWeightsPath)
            ModelDict = FaceModel.state_dict()
            # 1. Filter out unnecessary keys
            WeightsDict = {k: v for k, v in WeightsDict.items() if k in ModelDict}
            # 2. Overwrite entries in the existing state dict
            ModelDict.update(WeightsDict) 
            # 3. Load the new state dict
            FaceModel.load_state_dict(ModelDict)
            print('Done')

    if DataPath is not None and ModelWeightsPath is not None:
        return FaceData, FaceModel
    elif DataPath is not None:
        return FaceData
    else:
        return FaceModel


def GetFaceFromZSpace(FaceVectors, FaceModel):
    """Takes a list, tensor, or array as FaceVectors and returns the output of FaceModel given FaceVectors as input."""
    # Change FaceVectors to expected tensor of dim (1, Zdims)
    if type(FaceVectors) is not torch.Tensor:
        FaceVectors = torch.tensor(FaceVectors).float()
    if len(FaceVectors.shape) == 1:
        FaceVectors = FaceVectors.unsqueeze(0)
    if next(FaceModel.parameters()).is_cuda:
        FaceVectors = FaceVectors.cuda()

    return FaceModel(FaceVectors)


def ShowFace(FaceModel, FaceVectors=None, FaceData=None, Random=False, PrintGraph=True):
    """Displays an output face from a given StyleGAN model, FaceModel.
    
    Either a list, array, or tensor of FaceVectors can be specified, or the Random argument can be set True to generate a random face.

    Optional Arguments:
    FaceVectors -- List, array, or tensor of vectors representing inputs to FaceModel
    FaceData -- FaceLatents object to infer graph labels from
    Random -- Boolean flag specifying if a random face is to be generated; FaceVectors is ignored if True
    PrintGraph -- Boolean flag specifying if a graph representing basis faces is to be plotted
    """
    assert(not(FaceVectors is None and not Random)), 'If argument "Random" is set to False, FaceVectors must be provided'
    assert(not(PrintGraph and FaceData is None)), 'PrintGraph is set to True; FaceData object must be specified for graph to be plotted'
    assert(not(Random and FaceData is None)), 'Random is set to True; FaceData object must be specified for random face to be shown'

    if Random:
        FaceVectors = torch.abs(torch.randn(1, FaceData.Zdims))
        # FaceVectors /= FaceVectors.max()

    # Get model output
    if type(FaceVectors) is torch.Tensor:
        FaceVectors =  FaceVectors.detach()
    ModelOut = GetFaceFromZSpace(FaceVectors, FaceModel)

    # Convert FaceVectors to numpy array of dim (Zdims) for bar graph
    if PrintGraph:
        if FaceVectors is torch.Tensor:
            FaceVectors = FaceVectors.cpu()
        if type(FaceVectors) is not np.ndarray:
            FaceVectors = np.array(FaceVectors)
        if len(FaceVectors.shape) > 1:
            FaceVectors = FaceVectors.squeeze(0)
        # Plot bar graph of basis face weighting
        plt.bar([i for i in range(FaceData.Zdims)], FaceVectors)
        plt.xticks(range(FaceData.Zdims), FaceData.FaceNames, rotation=20)
        plt.grid()

    # Display output face
    ShowModelOutput(ModelOut, FigSize=6)

    # Return random face vectors
    if Random:
        return FaceVectors
    
def TransferLandmarks(FaceData, FaceModel):
    """Displays an output face from a given FaceData object containing landmarks as its Zspace and a FaceModel trained on landmarks"""
    # Loop through all stored landmarks
    for Z_i in range(len(FaceData.Zspace)):
        # Display stored face and landmarks
        ax = plt.subplot(1,2,1)
        ax.imshow(FaceData.Faces[Z_i].permute(1,2,0))
        Landmarks_x = FaceData.Zspace[Z_i,0::2].detach().cpu()
        Landmarks_y = FaceData.Zspace[Z_i,1::2].detach().cpu()
        ax.plot(Landmarks_x, Landmarks_y, '.')
        # Display model output given landmarks
        plt.subplot(1,2,2)
        LandmarksIn = FaceData.Zspace[Z_i].unsqueeze(0)
        # Send to CUDA if required
        if next(FaceModel.parameters()).is_cuda:
            LandmarksIn = LandmarksIn.cuda()
        ModelOut = FaceModel(LandmarksIn)
        # Transform model output to suitable form for plotting
        ModelOut = ModelOut.clamp_(-1, 1).add_(1).div_(2.0)
        ModelOut = ModelOut.detach().cpu()
        plt.imshow(ModelOut.squeeze().permute(1,2,0))
        plt.show()