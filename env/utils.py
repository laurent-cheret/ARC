import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def visualize_grids(env):
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    num_inputs = len(env.current_grids)
    num_memory = sum(1 for m in env.memory if m is not None)
    
    fig, axs = plt.subplots(1, num_inputs + num_memory, figsize=(5*(num_inputs + num_memory), 5))
    if num_inputs + num_memory == 1:
        axs = [axs]
    
    # Visualize current grids
    for i, grid_list in enumerate(env.current_grids):
        if grid_list:
            axs[i].imshow(grid_list[0], cmap=cmap, norm=norm)
            axs[i].set_title(f"Input {i+1}")
        else:
            axs[i].axis('off')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    # Visualize memory grids
    for i, memory_grids in enumerate(env.memory):
        if memory_grids is not None:
            axs[num_inputs + i].imshow(memory_grids[0], cmap=cmap, norm=norm)
            axs[num_inputs + i].set_title(f"Memory {i+1}")
        else:
            axs[num_inputs + i].axis('off')
        axs[num_inputs + i].set_xticks([])
        axs[num_inputs + i].set_yticks([])
    
    plt.tight_layout()
    plt.show()