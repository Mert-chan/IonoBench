
#############################################################
"""
myVisualFuns.py
@ Mert-chan 
@ 13 July 2025 (Last Modified)  
- Animations functions for visualizing prediction results and GIM data.
"""
#############################################################

# Libraries
#===============================================================================
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from source.myDataFuns import reverseHeliocentricSingle
#===============================================================================


# Default meshgrid for TEC maps (For reference)
#===============================================================================
# x = np.arange(-180, 180, 5)           # X axis (Longitude) Delete last element to get 72 elements
# y = np.arange(90, -90, -2.5)          # Y axis (Latitude) Add first element to get 72 elements 
# X, Y = np.meshgrid(x, y)              # Create a meshgrid
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
                        show_metrics=True,
                        save=False,
                        dpi=120,
                        save_dir=None 
                        ):
    """
    Create a styled TEC data animation matching the rectangular map style.
    Inputs:
        - data1: First data set (e.g., real VTEC).
        - data2: Second data set (e.g., predicted VTEC, Heliospheric shifted).
        - date_list: List of dates corresponding to the data.
        - titles: Dictionary with main title and subplot titles.
        - cmap: Colormap for the plots (default 'jet').
        - interval_ms: Interval between frames in milliseconds (default 300).
        - show_metrics: If True, display RMSE and R² metrics in the title (default True).
        - save: If True, save animation as GIF.
        - dpi: Resolution for saved GIF.
        - save_dir: Directory to save the GIF (default: ./visuals/).
    """
    plt.style.use('dark_background')

    # Generate lon/lat mesh
    lon = np.arange(-180, 180, 5)           # X axis (Longitude) Delete last element to get 72 elements
    lat = np.arange(90, -90, -2.5)          # Y axis (Latitude) Add first element to get 72 elements
    Lon, Lat = np.meshgrid(lon, lat)              # Create a meshgrid

    # Set color scale across all data
    vmin = np.nanmin([np.nanmin(data1), np.nanmin(data2)])
    vmax = np.nanmax([np.nanmax(data1), np.nanmax(data2)])

    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            gridspec_kw={'wspace': 0.15, 'hspace': 0.05},
                            constrained_layout=False)

    # Axis styling (ticks, coastlines, borders)
    def _style_axes(ax, show_y_labels=True):
        ax.set_global()
        ax.coastlines(resolution='110m', linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
        ax.gridlines(draw_labels=False, color='gray', linewidth=0.75, alpha=0.3)
        ax.set_xticks(np.arange(-150, 151, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}°' if x >= 0 else f'−{int(abs(x))}°'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}°' if y >= 0 else f'−{int(abs(y))}°'))
        ax.tick_params(axis='both', which='major', labelsize=11)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontweight('bold')

    for i, ax in enumerate(axs):
        _style_axes(ax, show_y_labels=(i == 0))

    # Initial frame
    im1 = axs[0].pcolormesh(Lon, Lat, data1[0], transform=ccrs.PlateCarree(),
                            cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    im2 = axs[1].pcolormesh(Lon, Lat, data2[0], transform=ccrs.PlateCarree(),
                            cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

    # Shared colorbar
    cb_ax = inset_axes(axs[-1], width="3%", height="110%", loc='center left',
                       bbox_to_anchor=(1.05, -0.1, 1, 1.2), bbox_transform=axs[-1].transAxes)
    cb = fig.colorbar(im2, cax=cb_ax, orientation='vertical')
    cb.set_label("TECU (10$^{16}$ e/m$^2$)", fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=12)
    cb.mappable.set_clim(vmin, vmax)

    # Main title with optional metrics
    main_title = f"{titles.get('main')}\n{date_list[0]:%Y-%m-%d %H:%M}"
    if show_metrics:
        rmse = np.sqrt(np.mean((data1[0] - data2[0]) ** 2))
        r2 = r2_score(data1[0].ravel(), data2[0].ravel())
        main_title += f"\nRMSE = {rmse:.4f}  |  R² = {r2:.4f}"
    main_title_obj = fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.95)

    # Static subplot titles
    axs[0].set_title(titles.get('subplot1', 'Real VTEC IGS'), pad=6, fontsize=12, fontweight='bold')
    axs[1].set_title(titles.get('subplot2', 'Predicted VTEC IGS'), pad=6, fontsize=12, fontweight='bold')

    # Frame update logic
    def _update(frame):
        z1, z2 = data1[frame], data2[frame]

        for i, ax in enumerate(axs):
            ax.clear()
            _style_axes(ax, show_y_labels=(i == 0))

        im1_new = axs[0].pcolormesh(Lon, Lat, z1, transform=ccrs.PlateCarree(),
                                    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        im2_new = axs[1].pcolormesh(Lon, Lat, z2, transform=ccrs.PlateCarree(),
                                    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

        if show_metrics:
            rmse = np.sqrt(np.mean((z1 - z2) ** 2))
            r2 = r2_score(z1.ravel(), z2.ravel())
            metrics_text = f"\nRMSE = {rmse:.4f}  |  R² = {r2:.4f}"
        else:
            metrics_text = ""

        date_str = date_list[frame].strftime('%Y-%m-%d %H:%M')
        main_title_obj.set_text(f"{titles.get('main', 'TEC Data Comparison')}\n{date_str}{metrics_text}")

        axs[0].set_title(titles.get('subplot1', 'Real VTEC IGS'), pad=6, fontsize=12, fontweight='bold')
        axs[1].set_title(titles.get('subplot2', 'Predicted VTEC IGS'), pad=6, fontsize=12, fontweight='bold')

        return [im1_new, im2_new]

    plt.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.85, wspace=0.15)

    # Create animation
    anim = animation.FuncAnimation(fig, _update,
                                   frames=len(date_list),
                                   interval=interval_ms,
                                   blit=False,
                                   repeat=True)

    # Show or save
    if not save:
        plt.show()
    else:
        plt.close(fig)
        if save_dir is None:
            save_dir = Path("visuals")
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = date_list[0].strftime('%Y%m%d_%H%M')
        title_key = titles.get('main', 'comparison').replace(" ", "_").replace(":", "").replace("/", "_")
        fname = f"{title_key}_{date_str}.gif"
        out_path = save_dir / fname
        print(f"Saving → {out_path}")
        anim.save(out_path, writer='pillow', dpi=dpi)

    return anim
#===============================================================================

# Function to create an animation comparing storm predictions with ground truth data using 
#=================================================================================
def spatialComparison_Anim(
    npz_path,
    dataDict,
    cfgs,
    n_start=0,
    n_end=None,
    horizon=6,
    interval_ms=300,
    save=False,
    dpi=120
):
    '''
    Creates an animation comparing storm predictions with ground truth data.
    Inputs:
        - npz_path: npz path to the file containing storm predictions and metadata.
        - dataDict: dictionary with minTEC and maxTEC for normalization.
        - cfgs: configuration object containing session name and base path.
        - n_start: starting index for storm sample (0-72, default 0).
        - n_end: ending index for storm sample (0-72, default None, uses all).
        - horizon: 0-11 steps: 0 → 2h, 5 → 12h, etc.
        - save: if True, saves the animation as a GIF.
        - dpi: resolution for the saved GIF.
    Returns:
        - anim: matplotlib animation object.
    '''
    data_npz = np.load(npz_path, allow_pickle=True)
    storm_meta = data_npz['storm_metadata'].item()
    storms = storm_meta['Storm Period']
    n_end = n_end or len(storms)

    # collect valid frames
    animation_data = []
    for n in range(n_start, min(n_end, len(storms))):
        try:
            dt = storms[n]
            preds = data_npz['full_period_predictions'][n + (12 - horizon)][horizon]
            labels = data_npz['full_period_labels'][n + (12 - horizon)][horizon]

            pred = reverseHeliocentricSingle(np.asarray(preds), dt)
            label = reverseHeliocentricSingle(np.asarray(labels), dt)
            res = pred - label

            rmse = np.sqrt(np.mean(res ** 2))
            r2 = r2_score(label.ravel(), pred.ravel())
            ssim_val = ssim(label, pred, data_range=dataDict['maxTEC'] - dataDict['minTEC'])

            animation_data.append({
                "date": dt,
                "label": label,
                "pred": pred,
                "residual": res,
                "rmse": rmse,
                "r2": r2,
                "ssim": ssim_val,
                "res_mean": res.mean(),
                "res_max": res.max(),
                "res_std": res.std()
            })
        except (IndexError, KeyError) as e:
            print(f"Skipping storm {n}: {e}")

    if not animation_data:
        raise ValueError("No valid frames collected.")

    print(f"Prepared {len(animation_data)} frames")

    # setup grid
    lon = np.arange(-180, 180, 5)           # X axis (Longitude) Delete last element to get 72 elements
    lat = np.arange(90, -90, -2.5)          # Y axis (Latitude) Add first element to get 72 elements
    Lon, Lat = np.meshgrid(lon, lat)              # Create a meshgrid
    # determine color ranges
    all_lbl = [f['label'] for f in animation_data]
    all_prd = [f['pred'] for f in animation_data]
    all_res = [f['residual'] for f in animation_data]

    vmin_data = np.min([d.min() for d in all_lbl + all_prd])
    vmax_data = np.max([d.max() for d in all_lbl + all_prd])
    max_abs_res = np.max([np.abs(d).max() for d in all_res])

    # layout
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 3, figsize=(20, 5),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.05})

    def _style_axes(ax, first=False):
        ax.set_global()
        ax.coastlines('110m', linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
        ax.gridlines(draw_labels=False, color='gray', linewidth=0.75, alpha=0.3)
        ax.set_xticks(np.arange(-150, 151, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}°' if x >= 0 else f'−{int(abs(x))}°'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}°' if y >= 0 else f'−{int(abs(y))}°'))
        ax.tick_params(axis='both', which='major', labelsize=11)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontweight('bold')

    for i, ax in enumerate(axs):
        _style_axes(ax, first=(i == 0))

    # initialize plots
    init = animation_data[0]
    im_data = axs[0].pcolormesh(Lon, Lat, init['label'],
                                cmap='jet', vmin=vmin_data, vmax=vmax_data,
                                shading='auto', transform=ccrs.PlateCarree())

    im_pred = axs[1].pcolormesh(Lon, Lat, init['pred'],
                                cmap='jet', vmin=vmin_data, vmax=vmax_data,
                                shading='auto', transform=ccrs.PlateCarree())

    im_res = axs[2].pcolormesh(Lon, Lat, init['residual'],
                               cmap='bwr', vmin=-max_abs_res, vmax=max_abs_res,
                               shading='auto', transform=ccrs.PlateCarree())

    # colorbars
    cb_ax1 = inset_axes(axs[1], width="4%", height="110%", loc='center left',
                        bbox_to_anchor=(1.05, -0.05, 1, 1.1),
                        bbox_transform=axs[1].transAxes)
    cb1 = fig.colorbar(im_data, cax=cb_ax1)
    cb1.set_label("TECU (10$^{16}$ e/m$^2$)", fontsize=12, fontweight='bold')
    cb1.ax.tick_params(labelsize=12)

    cb_ax2 = inset_axes(axs[2], width="4%", height="110%", loc='center left',
                        bbox_to_anchor=(1.05, -0.05, 1, 1.1),
                        bbox_transform=axs[2].transAxes)
    cb2 = fig.colorbar(im_res, cax=cb_ax2)
    cb2.set_label('Residual TECU', fontsize=12, fontweight='bold')
    cb2.ax.tick_params(labelsize=12)

    def update(idx):
        f = animation_data[idx]

        for i, ax in enumerate(axs):
            ax.clear()
            _style_axes(ax, first=(i == 0))

        axs[0].pcolormesh(Lon, Lat, f['label'],
                          cmap='jet', vmin=vmin_data, vmax=vmax_data,
                          shading='auto', transform=ccrs.PlateCarree())

        axs[1].pcolormesh(Lon, Lat, f['pred'],
                          cmap='jet', vmin=vmin_data, vmax=vmax_data,
                          shading='auto', transform=ccrs.PlateCarree())

        axs[2].pcolormesh(Lon, Lat, f['residual'],
                          cmap='bwr', vmin=-max_abs_res, vmax=max_abs_res,
                          shading='auto', transform=ccrs.PlateCarree())

        try:
            cs = axs[2].contour(Lon, Lat, f['residual'], levels=6,
                                colors='black', linewidths=0.8, linestyles='dotted',
                                transform=ccrs.PlateCarree())
            axs[2].clabel(cs, inline=True, fontsize=9, fmt='%.1f')
        except:
            pass

        htxt = f"{(horizon+1)*2} h"
        titles = [
            f'IGS Ground Truth \n{f["date"]:%Y-%m-%d %H:%M}',
            f'{cfgs.session.name} Prediction ({htxt})\n'
            f'RMSE={f["rmse"]:.2f}  R²={f["r2"]:.2f}  SSIM={f["ssim"]:.2f}',
            f'{cfgs.session.name} Residual ({htxt})\n'
            f'Max: {f["res_max"]:.2f}  Mean: {f["res_mean"]:.2f}  Std: {f["res_std"]:.2f}'
        ]
        for ax, t in zip(axs, titles):
            ax.set_title(t, pad=10, fontsize=14, fontweight='bold')

    anim = animation.FuncAnimation(fig, update,
                                   frames=len(animation_data),
                                   interval=interval_ms, blit=False)

    plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05, wspace=0.15)
    plt.close(fig)
    # extract storm number from filename
    storm_num = Path(npz_path).stem.split("_")[3] 
    if save:
        visuals_dir = Path(cfgs.paths.base_dir) / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{cfgs.session.name}_{(horizon+1)*2}h_storm{storm_num}_samples{n_start}-{n_end-1}.gif"
        out_path = visuals_dir / fname
        print(f"Saving GIF → {out_path} …")
        anim.save(out_path, writer='pillow', dpi=dpi)

    return anim
#=================================================================================