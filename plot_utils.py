from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import ortho_group
import torch

def PlotReduceRandOrtho(Z, Model, Annotate=True, ReturnW=False):
    """Plots distribution of Z and W given a tensor representing Z as input
    
    Reduces dimensions of Z and W to two dimensions by multiplying with a random orthogonal matrix
    """
    RandOrtho = ortho_group.rvs(512)
    RandOrtho = RandOrtho[:,0:2]

    with torch.no_grad():
       W = Model.g_all[0](Z)

    Z_np = Z.cpu().detach()
    W_np = W[:,0,:].cpu().detach()
    Z_reduced = np.dot(Z_np, RandOrtho)
    W_reduced = np.dot(W_np, RandOrtho)

    plt.figure(figsize=(5,5))
    colors = cm.rainbow(np.linspace(0, 1, 2))
    plt.scatter(Z_reduced[:,0], Z_reduced[:,1], color = colors[0])
    plt.scatter(W_reduced[:,0], W_reduced[:,1], color = colors[1])
    plt.legend(['Z','W'])

    if Annotate:
        for i in list(range(len(Z_reduced))):
            plt.annotate(i, (Z_reduced[i,0], Z_reduced[i,1]))
            plt.annotate(i, (W_reduced[i,0], W_reduced[i,1]))

    if ReturnW:
        return W


def ShowModelOutput(ModelOut, FigSize=4):
    """Takes a tensor produced from StyleGAN as input and displays it as an image"""
    # Clamp and normalise tensor if not within expected range of [0,1]
    if not(ModelOut.max() <= 1 and ModelOut.min() >= 0):
        ModelOut = ModelOut.clamp_(-1, 1).add_(1).div_(2.0)

    BatchSize = ModelOut.shape[0]

    plt.subplots(1, BatchSize, figsize=((FigSize+0.5)*BatchSize, FigSize))
    for Sample_i in range(BatchSize):
        plt.subplot(1, BatchSize, Sample_i+1)
        Img = ModelOut[Sample_i].detach().squeeze(0).cpu().permute(1, 2, 0).numpy()
        plt.imshow(Img)
    plt.show()


def ShowModelOutputFromInput(Input, Model, W_input=False):
    # Transform input if necessary and use only synthesis network
    # if using transformed latent vectors, W as input
    if W_input:
        if len(Input.shape) < 3:
            Input = Input.unsqueeze(1).repeat(1,18,1)
        Model = Model.g_all[1]

    # Get output of model, transform to valid values, and display
    with torch.no_grad():
        ModelOut = Model(Input)
    ModelOut = ModelOut.clamp_(-1, 1).add_(1).div_(2.0)
    ShowModelOutput(ModelOut)


def InterpolateW_extend(FixedLatents, ModelSynth, Device):
    import IPython
    # Get a few Images
    Latents = []
    for i in range(len(FixedLatents)-1):
        Latents.append(FixedLatents[i] + (FixedLatents[i + 1] - FixedLatents[i]) * \
                       torch.arange(0, 1, 0.05, device=Device).view(-1,1,1))
    Latents.append(FixedLatents[-1])
    Latents = torch.cat(Latents, dim=0)

    with torch.no_grad():
        for Latent in Latents:
            Latent = Latent.to(Device)
            img = ModelSynth(Latent.unsqueeze(0))
            img = img.clamp_(-1, 1).add_(1).div_(2.0)        
            img = img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()
            plt.figure(figsize=(8,8))
            plt.imshow(img)
            IPython.display.clear_output(True)
            plt.show()


def InterpolateCompressed(FixedLatents, Model, Steps=20, ClearOutput=True, FigSize=(8,8), Tiled=False):
    """Display interpolation between a list of fixed latents in the Zspace"""
    import IPython
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Add batch dimension if needed
    if len(FixedLatents[0].shape) == 1:
        FixedLatents = [Latent.unsqueeze(0) for Latent in FixedLatents]

    # Get a few Images
    Latents = []
    for i in range(len(FixedLatents)-1):
        Latents.append(FixedLatents[i] + (FixedLatents[i + 1] - FixedLatents[i]) * \
                       torch.arange(0, 1, 1/Steps, device=device).view(-1,1))
    Latents.append(FixedLatents[-1])
    Latents = torch.cat(Latents, dim=0)

    if Tiled:
        FigCols = 8
        TotalFigs = (len(FixedLatents) - 1) * Steps + 1
        FigRows = TotalFigs//FigCols if TotalFigs % FigCols == 0 else TotalFigs//FigCols + 1
        fig, ax = plt.subplots(FigRows, FigCols, figsize=(FigSize[0]*FigCols, FigSize[1]*FigRows))
    with torch.no_grad():
        for i, Latent in enumerate(Latents):
            Latent = Latent.to(device)
            img = Model(Latent.unsqueeze(0))
            img = img.clamp_(-1, 1).add_(1).div_(2.0)        
            img = img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()

            if not Tiled:
                plt.figure(figsize=FigSize)
                plt.imshow(img)
                if ClearOutput:
                    IPython.display.clear_output(True)
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.show()
            else:
                plt.figure(fig.number)
                plt.subplot(FigRows, FigCols, i+1)
                plt.imshow(img)
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
        plt.show()