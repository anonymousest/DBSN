import torch
import matplotlib.pyplot as plt
import math

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def _ada_gumbel_softmax(logits, scales, tau=1., eps=1e-10):

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)

    returns = []
    for scale in scales:
        returns.append(((gumbels*scale + logits)/tau).softmax(-1).data.numpy())

    return returns

if __name__ == "__main__":
    logits = torch.Tensor([0.05, 0.05, 0.5, 0.4]).log()
    scales = [1.0, 0.5, 0.3, 0.1, 0] #[0., 0.1, 0.3, 0.5, 1.0]

    samples = []
    for _ in range(5):
        samples.append(_ada_gumbel_softmax(logits, scales))

    fig, axs = plt.subplots(5, len(scales), figsize=(9, 3), sharey=True, sharex=True)
    for i in range(5):
        for j in range(len(scales)):
            print(i, j, samples[i][j])
            axs[i][j].bar(range(logits.size()[0]), samples[i][j], color=tableau20[i])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            if j == 0:
                axs[i][j].set_ylabel("sample#" + str(i), {'family' : 'normal', 'size'   : 6})
            if i == 4:
                axs[i][j].set_xlabel("$\mathregular{\\beta_{(i,j)}}$=" + str(scales[j]))

    #fig.suptitle('Plotting')
    plt.savefig("adacon.pdf")
