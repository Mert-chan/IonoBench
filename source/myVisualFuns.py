

# Libraries
#############################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.metrics import r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Default meshgrid for TEC maps
#===============================================================================
x = np.arange(-180, 180, 5)           # X axis (Longitude) Delete last element to get 72 elements
y = np.arange(90, -90, -2.5)        # Y axis (Latitude) Add first element to get 72 elements
X, Y = np.meshgrid(x, y)              # Create a meshgrid
#===============================================================================

# Function to compare two TEC data sets and create an animation
#===============================================================================
def makeComparison_Anim(
                        data1, 
                        data2, 
                        date_list, 
                        titles,
                        cmap='jet',
                        interval_ms=300,
                        residuals=False, 
                        show_metrics=True
                        ):
    """
    Create a styled TEC data animation matching the rectangular map style.
    """
    plt.style.use('dark_background')
    # Coordinates
    _ , lat_size, lon_size = data1.shape
    lon = np.linspace(-180, 180, lon_size, endpoint=False)
    lat = np.linspace(90, -90, lat_size)
    Lon, Lat = np.meshgrid(lon, lat)
    
    vmin_pred, vmax_pred = np.min(data1), np.max(data2)
    
    # Plot settings
    plt.rcParams.update({'font.size': 15})  # Increase base font size
    fig, axs = plt.subplots(1, 2, figsize=(12, 4),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            gridspec_kw={'wspace': 0.15, 'hspace': 0.05},
                            constrained_layout=False)
    
    # Calculate color scale
    vmin = np.nanmin([data1, data2])
    vmax = np.nanmax([data1, data2])
    
    main_title = f"{titles.get('main')}\n{date_list[0]:%Y-%m-%d %H:%M}"
    if show_metrics:
        rmse = np.sqrt(np.mean((data1[0] - data2[0]) ** 2))
        r2 = r2_score(data1[0].ravel(), data2[0].ravel())
        main_title += f"\nRMSE = {rmse:.4f}  |  R² = {r2:.4f}"
    main_title_obj = fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.95)
    
    # Store title objects for updating
    title1 = axs[0].set_title(titles.get('subplot1', 'Real VTEC IGS'), pad=6, fontsize=12, fontweight='bold')
    title2 = axs[1].set_title(titles.get('subplot2', 'Predicted VTEC IGS'), pad=6, fontsize=12, fontweight='bold')
    
    cmap, vmin, vmax = 'jet', vmin_pred, vmax_pred
    data_list = [data1, data2]
    ims = []
    
    for i, ax in enumerate(axs):
        ax.set_global()
        ax.coastlines(resolution='110m', linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
        ax.gridlines(draw_labels=False, color='gray', linewidth=0.75, alpha=0.3)

        if i < len(data_list):
            plot_data = data_list[i][0]
            if i == 0:
                im = ax.pcolormesh(Lon, Lat, plot_data, transform=ccrs.PlateCarree(),
                                cmap='jet', shading='auto', vmin=vmin_pred, vmax=vmax_pred)
            else:
                im = ax.pcolormesh(Lon, Lat, plot_data, transform=ccrs.PlateCarree(),
                                cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
            ims.append(im)
        
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}°'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}°'))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # make ticks bold
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontweight('bold')
    
    cb_height = "110%"
    cb_ax = inset_axes(axs[-1], width="3%", height=cb_height, loc='center left',
                         bbox_to_anchor=(1.05, -0.1, 1, 1.2), bbox_transform=axs[-1].transAxes)
    cb = fig.colorbar(ims[-1], cax=cb_ax, orientation='vertical')
    cb.set_label("TECU (10$^{16}$ e/m$^2$)", fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=12)

    def _update(frame):
        """Update function for animation frames"""
        z1, z2 = data1[frame], data2[frame]
        im1, im2 = ims[0], ims[1]
        
        # Update image data
        im1.set_array(z1.ravel())
        im2.set_array(z2.ravel())
        cb.update_normal(ims[1])
              
        # Calculate metrics if requested
        metrics_text = ""
        if show_metrics:
            rmse = np.sqrt(np.mean((z1 - z2) ** 2))
            r2 = r2_score(z1.ravel(), z2.ravel())
            metrics_text = f"\nRMSE = {rmse:.4f}  |  R² = {r2:.4f}"
        
        # Update titles
        date_str = date_list[frame].strftime('%Y-%m-%d %H:%M')
        
        # Main title
        main_title_obj.set_text(f"{titles.get('main', 'TEC Data Comparison')}\n{date_str}{metrics_text}")
        
        # Subplot titles (keep them static)
        title1.set_text(titles.get('subplot1', 'Real VTEC IGM'))
        title2.set_text(titles.get('subplot2', 'Longitude-Shifted VTEC IGM'))
        
        return ims[0], ims[1], cb.ax.collections[0], main_title_obj
    
    plt.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.85, wspace=0.15)
    # Create animation
    anim = animation.FuncAnimation(fig, _update,
                                   frames=len(date_list),
                                   interval=interval_ms,
                                   blit=False,
                                   repeat=True)
    plt.close(fig)  # Close the figure to prevent display in Jupyter
    return anim  
#############################################################


#