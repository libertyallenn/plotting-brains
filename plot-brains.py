import os
import os.path as op
from os.path import join
from os import makedirs

os.environ.pop("MPLBACKEND", None) # Remove any preset backend
import numpy as np # numpy is short for "numerical python" and it does math
import nibabel as nib # nibabel handles nifti images
from nilearn import datasets
from nilearn.reporting import get_clusters_table
from nilearn.maskers import NiftiMasker
from nilearn.image import resample_to_img, threshold_img, index_img, math_img
from nilearn.plotting import plot_stat_map
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # gridspec helps us put lots of panels on one figure
from matplotlib.colors import ListedColormap # LinearSegmentedColormap
from neuromaps import transforms
from neuromaps.datasets import fetch_fslr
from surfplot import Plot 
from gradec.utils import _zero_medial_wall


def main(): 
    data_dir = "./data"
    out_dir = "./figures"
    fig_save_dir = out_dir
    makedirs(fig_save_dir, exist_ok=True)

    CMAP = matplotlib.colormaps["Spectral_r"]
    template = datasets.load_mni152_template(resolution=1)
    mask = datasets.load_mni152_brain_mask(resolution=1)
    # first_img = nib.load(filename[0])
    filenames = [    
        join(data_dir, "k_cluster_maps_k3_group0_tstat_fdr_clust_p-01_vox-50.nii.gz"),
        join(data_dir, "k_cluster_maps_k3_group1_tstat_fdr_clust_p-01_vox-50.nii.gz"),
        join(data_dir, "k_cluster_maps_k3_group2_tstat_fdr_clust_p-01_vox-50.nii.gz"),
        join(data_dir, "k_cluster_maps_k4_group0_tstat_fdr_clust_p-01_vox-50.nii.gz"),
        join(data_dir, "k_cluster_maps_k4_group1_tstat_fdr_clust_p-01_vox-50.nii.gz"),
        join(data_dir, "k_cluster_maps_k4_group2_tstat_fdr_clust_p-01_vox-50.nii.gz"),
        join(data_dir, "k_cluster_maps_k4_group3_tstat_fdr_clust_p-01_vox-50.nii.gz")]
    print("Template affine:\n", template.affine)
    print("Template shape", template.shape)
    print("Mask affine:\n", mask.affine)
    print("Mask shape:", mask.affine)
    z_thresh = 0.00001
    masker = None

    def plot_vol(
        nii_img_thr, threshold, mask_contours=None, vmax=6, alpha=1, cmap=CMAP, dim=-0.45):
        print("Stat map affine:\n", nii_img_thr.affine)
        print("Stat map shape:", nii_img_thr.shape)
        template = datasets.load_mni152_template(resolution=1)
        display_modes = ["x", "y", "z"]
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(2, 2, figure=fig)
        
        for dsp_i, display_mode in enumerate(display_modes):
            if display_mode == "z":
                ax = fig.add_subplot(gs[:, 1], aspect="equal")
                colorbar = True
            else:
                ax = fig.add_subplot(gs[dsp_i, 0], aspect="equal")
                colorbar = False
            display = plot_stat_map(
                nii_img_thr,
                bg_img=template,
                black_bg=False,
                draw_cross=False,
                annotate=False,
                alpha=alpha,
                cmap=cmap,
                threshold=threshold,
                colorbar=colorbar,
                display_mode=display_mode,
                cut_coords=1,
                vmax=vmax,
                axes=ax,
                dim=dim, 
                interpolation='hanning'
            )
            display.annotate(size=7)
            ax.set_title(ax.get_title(), fontsize=6)
            for txt in ax.texts:
                txt.set_fontsize(6)
                x, y = txt.get_position()
                txt.set_position((x,y- 0.02))
            if mask_contours is not None:
                nii_mask_smooth = math_img("smooth)img(img, 2)", img=nii_thr_img)
                display.add_contours(nii_mask_smooth, levels=[0.5], colors="black")
        return fig
    
    for i, filename in enumerate(filenames):
        print(f"Processing, {filename}...")
        # first_img = nib.load(filename[0])
        nii_img = nib.load(filename)
        nii_thr_img = threshold_img(nii_img, threshold=z_thresh, two_sided=True)
        mask_resampled = resample_to_img(mask, nii_thr_img, interpolation='nearest')
        masker = NiftiMasker(mask_img=mask_resampled).fit()
        nii_thr_arr = masker.transform(nii_thr_img)
        nii_contour_arr = np.zeros_like(nii_thr_arr)
        nii_contour_arr[(nii_thr_arr > z_thresh) | (nii_thr_arr < -z_thresh)] = 1
        nii_contour_img = masker.inverse_transform(nii_contour_arr)
        nii_contour_img_3d = index_img(nii_contour_img, 0)
        vmax = round(np.max(np.abs(nii_thr_arr)), 2)
        vmax = 13 if vmax > 13 else vmax
        fig = plot_vol(nii_img,threshold=z_thresh)
        fig.savefig(op.join(fig_save_dir, f"{op.basename(filename).replace('.nii.gz','')}_volume.png"), dpi=3000, bbox_inches='tight')
        plt.close(fig)

        def trim_image(img=None, tol=1, fix=True):
            if fix:
                mask = img != tol
            else:
                mask = img <= tol
            if img.ndim == 3:
                mask = mask.any(2)
                mask0, mask1 = mask.any(0), mask.any(1)
                return img[np.ix_(mask1, mask0)]
            else: 
                mask0, mask1=mask.any(0),mask.any(1)
            return img[np.ix_(mask1, mask0)]

        def plot_surf(nii_img_thr, mask_contours=None, vmax=8, cmap=CMAP, alpha=1.0):
            map_lh, map_rh = transforms.mni152_to_fslr(nii_img_thr, fslr_density="32k")
            map_lh, map_rh = _zero_medial_wall(map_lh, map_rh, space="fsLR", density="32k")
            surfaces = fetch_fslr(density="32k")
            lh, rh = surfaces["inflated"]
            sulc_lh, sulc_rh = surfaces["sulc"]
            p = Plot(surf_lh=lh, surf_rh=rh, layout="grid")
            p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
            p.add_layer({"left": map_lh, "right": map_rh}, cmap=cmap, cbar=True, color_range=(-vmax, vmax), alpha=1.0)
            if mask_contours:
                mask_lh, mask_rh = transforms.mni152_to_fslr(mask_contours, fslr_density="32k")
                mask_lh, mask_rh = _zero_medial_wall(mask_lh, mask_rh, space="fsLR", density="32k")      
                mask_arr_lh = mask_lh.agg_data()
                mask_arr_rh = mask_rh.agg_data()
                contours_lh = np.zeros_like(mask_arr_lh)
                contours_lh[mask_arr_lh != 0] = 1
                contours_rh = np.zeros_like(mask_arr_rh)
                contours_rh[mask_arr_rh != 0] = 1
                colors = [(0, 0, 0, 0)]
                contour_cmap = ListedColormap(colors, "regions", N=1)
                # line_cmap = ListedColormap(["black"], "regions", N=1)
                # p.add_layer({"left": contours_lh, "right": contours_rh}, cmap=line_cmap, as_outline=True, cbar=False)
                p.add_layer({"left": contours_lh, "right": contours_rh}, cmap=contour_cmap, cbar=False)
            return p.build()

    # Volume plot
    vol_fig = plot_vol(nii_thr_img, z_thresh, vmax=vmax, cmap=CMAP)
    vol_fig.savefig(
    op.join(fig_save_dir, f"{op.basename(filename).replace('.nii.gz','')}_volume.png"), dpi=3000, bbox_inches='tight')
    plt.close(vol_fig)     
    # Surface plot
    surf_fig = plot_surf(nii_thr_img, mask_contours=nii_contour_img_3d, vmax=vmax, cmap=CMAP)
    surf_fig.savefig(
    op.join(fig_save_dir, f"{op.basename(filename).replace('.nii.gz','')}_surface.png"), dpi=3000, bbox_inches='tight')   
    plt.close(surf_fig)

    # Generate cluster table and save thresholded images
    clusters = get_clusters_table(nii_thr_img, z_thresh, two_sided=True)   
    print("Significant clusters:")
    print(clusters)
    clusters.to_csv(op.join(fig_save_dir, "significant_clusters_table.csv"), index=False)

if __name__ == "__main__":
    main()
