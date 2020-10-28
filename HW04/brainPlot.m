function [] = brainPlot(z_mat)
a = make_nii(z_mat);
save_nii(a,'temp.nii');
[~,~]=convertnii2mat('temp.nii','untouch');
delete ('temp.nii')
