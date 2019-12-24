from get_features import MakeFeatureModel, GetFeatures
from crop_align_face import LandmarksDetector, unpack_bz2
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from plot_utils import ShowModelOutput
import numpy as np
import os
import time
import pickle
import functools

class FaceLatents:
    def __init__(self, Zdims=None, Trainable=True, Size=8, Zspace=None, Wspace=None, UseCuda=True):
        """Initialise FaceLatents object with latent tensors of input to encoder Z and extended latent space W+ of StyleGAN
        
        Attributes:
        Zspace -- Latent input to encoder with compressed dimensionality, dimension (Size, Zdims)
        Wspace -- Extended latent input to AdaIn layers of StyleGAN's synthesis network, dimension (Size, 18, 512)
        Size -- Number of basis faces stored
        Faces -- Tensor representations of each face, dimension (Size, 3, 1024, 1024)
        FaceNames -- List of names of each stored face
        Zdims -- Dimensionality of Zspace
        FeatureMaps -- List of feature maps; each feature map has dimension (Size, C, H, W) where C,H,W vary across maps
        Trainable -- Boolean flag specifying if Zspace and Wspace should be trainable, i.e. requires_grad
        device -- Device PyTorch tensors are stored on (uses CUDA if available)
        """

        if UseCuda:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cpu':
                raise Warning('UseCuda was set true but cuda is not available, using cpu')
        else:
            self.device = 'cpu'

        # Infer size of data (number of basis faces) based on Zspace and Wspace in that order of priority
        if Zspace is not None:
            print('Size was specified a value of %d but overwritten by given Zspace dimension of '%Size, len(Zspace))
            Size = len(Zspace)
        elif Wspace is not None:
            print('Size was specified a value of %d but overwritten by given Wspace dimension of '%Size, len(Wspace))
            Size = len(Wspace)

        if Zspace is not None and Wspace is not None:
            assert(not(len(Zspace) == len(Wspace))), 'Dimensions of Zspace and Wspace do not match'
        if Wspace is not None:
            assert(Wspace.shape[1] == 18 and Wspace.shape[2] == 512), 'Latent vectors of W must be (Size, 18, 512)'

        # Construct the input space of compressed dimensionality, Zspace
        if Zspace is not None:
            self.Zspace = Zspace.to(self.device)
        elif Zdims is None or Zdims == Size:
            print('Using default Zspace - identity matrix (orthogonal vectors) of dim ', Size)
            self.Zspace = torch.eye(Size).to(self.device)
        else:
            self.Zspace = torch.zeros(Size, Zdims).to(self.device)
        
        # Construct the extended latent Wspace to be learned via training
        if Wspace is not None:
            self.Wspace = Wspace.to(self.device)
        else:
            print('Using default Wspace - zero matrix of dim ', Size)
            self.Wspace = torch.zeros(Size,18,512).to(self.device)

        if Trainable:
            self.Trainable = True
            self.Zspace.requires_grad = True
            self.Wspace.requires_grad = True
        else:
            self.Trainable = False

        self.Size = Size    # Number of basis faces (if basis face encoding is being done)
        self.Faces = None   # Tensor representation of faces
        self.FacePaths = [] # List of paths for faces
        self.FaceNames = [] # List of names of stored faces
        self.Zdims = self.Zspace.shape[1]   # Dimensionality of input Zspace
        self.FeatureMaps = []   # Feature maps of basis faces

    def GetFacesFromPaths(self, TargetPaths, PrintFaces=True, PrintInterval=1):
        """Stores tensor representations of face images given a list of paths"""

        if not(len(TargetPaths) == self.Size):
            print('Warning: Number of target images provided does not equal number of stored latents')
        
        # Get names of faces from file name
        self.FacePaths = TargetPaths
        self.FaceNames = [os.path.splitext(os.path.basename(Path))[0] for Path in self.FacePaths]

        # Get tensors of faces and plot
        self.Faces = torch.zeros(len(TargetPaths), 3, 1024, 1024).to(self.device)
        for i, Path in enumerate(TargetPaths):
            # Scale pixel values to [0,1]
            Face = np.array(Image.open(Path))/255.0
            if PrintFaces and i % PrintInterval==0:
                ax = plt.subplot(1, 1, 1)
                ax.set_title(self.FaceNames[i])
                ax.imshow(Face)
                plt.show()
                
            # Convert to tensor and arrange to C,H,W
            Face = torch.from_numpy(Face).float().permute((2,0,1))
            self.Faces[i] = Face
        # Reshape Wspace based on number of obtained faces
        self.Wspace = torch.zeros(len(TargetPaths),18,512).to(self.device)
        # Set gradient flag if trainable
        if self.Trainable:
            self.Zspace.requires_grad = True
            self.Wspace.requires_grad = True
        print('Reshaped Wspace to match number of stored faces, Wspace dim: ', self.Wspace.shape)


    def GetLandmarks(self, LandmarksModelPath='landmarks/shape_predictor_68_face_landmarks.dat', PrintFaces=True, PrintInterval=1):
        """Obtains landmarks of stored faces and stores them as the input Zspace for pose conditioning"""

        # Initialise landmarks detector
        if not os.path.isfile(LandmarksModelPath):
            print('Landmarks model file does not exist at: ', LandmarksModelPath)
            print('Downloading from dlib directory...')
            from keras.utils import get_file
            LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            LandmarksModelPath = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))

        landmarks_detector = LandmarksDetector(LandmarksModelPath)
        
        # Get new landmarks from aligned images
        AllFaceLandmarks = []
        DeleteIndices = []
        for i, Path in enumerate(self.FacePaths):
            AlignedLandmarks = [landmark for landmark in landmarks_detector.get_landmarks(Path)]
            try:
                assert(not(len(AlignedLandmarks) > 1)), ('More than one face detected in "%s"' %Path)
            except:
                print('This path will be deleted')
                DeleteIndices.append(i)
                continue
            
            try:
                AlignedLandmarks = AlignedLandmarks[0]
            except:
                print("No face detected for ", Path, "\n This path will be deleted.")
                DeleteIndices.append(i)
                continue

            # Plot faces with landmarks
            if PrintFaces and i % PrintInterval == 0:
                ax = plt.subplot(1, 1, 1)
                ax.set_title(self.FaceNames[i])
                Face = np.array(Image.open(Path))/255.0
                ax.imshow(Face)
                Landmarks_x = [AlignedLandmarks[j][0] for j in range(len(AlignedLandmarks))]
                Landmarks_y = [AlignedLandmarks[j][1] for j in range(len(AlignedLandmarks))]
                ax.plot(Landmarks_x, Landmarks_y, '.')
                plt.show()

            # Flatten list of tuples and append to all stored landmarks
            FaceLandmarks = [Coordinates for Landmark in AlignedLandmarks for Coordinates in Landmark]
            AllFaceLandmarks.append(FaceLandmarks)

        # Delete faces with no face detected
        Deleted = 0
        for Idx in DeleteIndices:
            ActualDeleteIdx = Idx - Deleted
            del self.FacePaths[ActualDeleteIdx]
            self.Faces = torch.cat((self.Faces[0:ActualDeleteIdx], self.Faces[ActualDeleteIdx+1:]), dim=0)
            Deleted += 1

        # Convert list of lists to array
        AllFaceLandmarks = np.array(AllFaceLandmarks)

        # Save face landmarks in Zspace
        self.Zspace = torch.from_numpy(AllFaceLandmarks).float().to(self.device)
        # Update Size and reshape Wspace based on number of obtained faces
        self.Size = self.Zspace.shape[0]
        self.Wspace = torch.zeros(self.Size,18,512).to(self.device)
        # Set gradient flag if trainable
        if self.Trainable:
            self.Zspace.requires_grad = True
            self.Wspace.requires_grad = True
        print('Reshaped Wspace to match number of faces with landmarks, Wspace dim: ', self.Wspace.shape)
                

    def GetFeatureMaps(self):
        """Stores feature maps of the stored faces obtained using a pretrained vgg16 network"""

        # Make feature model
        FeatureModel = MakeFeatureModel(modelName='vgg16')

        # Get list of list of feature maps
        FeatureMaps = []
        for Face in self.Faces:
            Face = Face.unsqueeze(0)
            Face = nn.functional.interpolate(Face, size=(256,256), mode='bilinear')
            FeatureMaps.append(GetFeatures(Face, FeatureModel))
        assert(len(FeatureMaps) == len(self.Faces)), 'Feature maps obtained not equal to number of stored faces'

        # Concatenate feature maps in batch size dimension
        self.FeatureMaps = []
        for i in range(len(FeatureMaps[0])):
            self.FeatureMaps.append(torch.cat([Map[i] for Map in FeatureMaps], dim=0))


def TrainLatentsEnc(Model, FaceLatents, Epochs_n, LearnRate, LearnRateW, BatchSize=2,\
                    Weight_feature=1e2, Weight_pixel=1, Weight_tp=1e4, \
                    TrainZ=True, StoredTargetFeatures=True, PrintInterval=20, MaxBatchPrint=10, UseCuda=True):
    """Training function to train a StyleGAN encoder network and optionally an input Zspace to map the Zspace 
    to an extended Wspace (input to StyleGAN synthesis network) corresponding to target faces stored in FaceLatents.
    """

    # Initialisation of model
    if UseCuda:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            raise Warning('UseCuda was set true but cuda is not available, using cpu')
    else:
        device = 'cpu'

    if device != FaceLatents.device:
        print('Training function and FaceLatents class are using different devices')
    Model = Model.to(device)
    for param in Model.parameters():
        param.requires_grad = False

    # Initialise learnable parameters
    OptimParams = []
    # Input Zspace
    if TrainZ:
        OptimParams = OptimParams + [FaceLatents.Zspace]
    # Parameters of encoder network
    DecompParams = list(Model.g_all.g_decompress.parameters())
    for params in DecompParams:
        params.requires_grad = True
    OptimParams = OptimParams + DecompParams
    # Initialise optimiser
    Optimiser = torch.optim.Adam(OptimParams, lr=LearnRate)

    # Initialise optimiser for W+ for target propagation
    WOptimiser = torch.optim.Adam([FaceLatents.Wspace], lr=LearnRateW)

    # Initialise loss variables
    Loss = torch.tensor(0, device=device).float()
    Loss_feature = torch.tensor(0, device=device).float()
    Loss_pixel = torch.tensor(0, device=device).float()
    Loss_tp = torch.tensor(0, device=device).float()

    # Initialise feature model to get feature maps from
    if Weight_feature > 0:
        FeatureModel = MakeFeatureModel(modelName='vgg16')

    # Calculate batches
    Batch_n = FaceLatents.Size // BatchSize
    if FaceLatents.Size % BatchSize is not 0:
        Batch_n += 1

    # Training loop
    TimeStart = time.time()
    for Epoch_i in range(1, Epochs_n+1):
        # Shuffle batch indices
        RandIdx = list(np.random.permutation(FaceLatents.Size))

        for Batch_i in range(Batch_n):
            # Reset gradients
            Optimiser.zero_grad()
            WOptimiser.zero_grad()

            # Get a batch
            StartIdx = Batch_i*BatchSize
            EndIdx = StartIdx + BatchSize
            BatchIdx = RandIdx[StartIdx:EndIdx]
            Zspace = FaceLatents.Zspace[BatchIdx].to(device)
            Wspace = FaceLatents.Wspace[BatchIdx].to(device)
            Faces = FaceLatents.Faces[BatchIdx].to(device)
            
            # Get target features for the batch
            if StoredTargetFeatures:
                TargetFeatures = [Maps[BatchIdx].to(device) for Maps in FaceLatents.FeatureMaps]
            else:
                Faces_ds = nn.functional.interpolate(Faces, size=(256,256), mode='bilinear')
                TargetFeatures = GetFeatures(Faces_ds, FeatureModel, UseCuda=UseCuda)

            # Learn target W+ and compute target propagation loss
            if Weight_tp > 0:
                # Get synthesis output of W+ target
                ModelOut_tgt = Model.g_all.g_synthesis(Wspace)

                # Standardise pixel values to [0,1]
                ModelOut_tgt = ModelOut_tgt.clone().clamp_(-1, 1).add_(1).div_(2.0)

                # Target feature loss
                if Weight_feature > 0:
                    # Get feature maps
                    ModelOut_ds_tgt = nn.functional.interpolate(ModelOut_tgt, size=(256,256), mode='bilinear')
                    OutFeatures_tgt = GetFeatures(ModelOut_ds_tgt, FeatureModel)
                    Loss_feature_tgt = functools.reduce(
                                            lambda x, y : x + y , 
                                            [nn.functional.mse_loss(Features[0], Features[1])\
                                            for Features in zip(OutFeatures_tgt, TargetFeatures)]) 
                
                # Target pixel loss
                Loss_pixel_tgt = nn.functional.mse_loss(ModelOut_tgt, Faces)
                # Aggregate pixel and feature loss
                Loss_W_ex_tgt = Weight_feature*Loss_feature_tgt + Weight_pixel*Loss_pixel_tgt
                # Learn W+ target
                Loss_W_ex_tgt.backward()
                WOptimiser.step()

                # Calculate target propagation loss
                Wspace_Z = Model.g_all.g_decompress(Zspace)
                Loss_tp = nn.functional.mse_loss(Wspace_Z, Wspace.detach())

            # Get model output from Zspace
            ModelOut = Model(Zspace)

            # Standardise output pixel values to [0,1]
            ModelOut = ModelOut.clone().clamp_(-1, 1).add_(1).div_(2.0)

            # Compute feature (perceptual) loss
            Loss_feature = torch.tensor(0, device=device).float()
            if Weight_feature > 0:
                ModelOut_ds = nn.functional.interpolate(ModelOut, size=(256,256), mode='bilinear')
                OutFeatures = GetFeatures(ModelOut_ds, FeatureModel, UseCuda=UseCuda)
                Loss_feature = functools.reduce(
                                    lambda x, y : x + y , 
                                    [nn.functional.mse_loss(Features[0], Features[1])\
                                    for Features in zip(OutFeatures, TargetFeatures)]) 
            
            # Compute pixel loss
            Loss_pixel = nn.functional.mse_loss(ModelOut, Faces)
                
            # Aggregate losses
            Loss = Weight_feature*Loss_feature + Weight_pixel*Loss_pixel + Weight_tp*Loss_tp
            
            # Print metrics
            if (Epoch_i == 1 or Epoch_i == Epochs_n or Epoch_i % PrintInterval == 0) \
                and Batch_i < MaxBatchPrint:
                print('Epoch: ', Epoch_i)
                print('Batch: ', Batch_i + 1)
                print('This batch: ', [FaceLatents.FaceNames[Idx] for Idx in BatchIdx])
                print('Total time elapsed: ', time.time()-TimeStart, ' s')
                print('Feature loss: ', Loss_feature.item(), \
                      ' | Pixel loss: ', Loss_pixel.item(), \
                      ' | Target prop loss: ', Loss_tp.item())
                print('Weighted - Feature loss: ', Weight_feature*Loss_feature.item(), \
                      ' | Pixel loss: ', Weight_pixel*Loss_pixel.item(), \
                      ' | Target prop loss: ', Weight_tp*Loss_tp.item())
                print('Total loss: ', Loss.item())
                ShowModelOutput(ModelOut)
                if Weight_tp > 0:
                    print('Target W+ output')
                    print('W+ loss: ', Loss_W_ex_tgt.item())
                    ShowModelOutput(Model.g_all.g_synthesis(Wspace))

            # Perform gradient descent and backprop
            Loss.backward()
            Optimiser.step()