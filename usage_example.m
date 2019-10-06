clear variables; close all;
img_1 = imread('11.jpg');
img_2 = imread('12.jpg');

[output_image, overlap_image, residue, num_inliers, num_matches] = ImStich(img_1, img_2, 'SIFT',3 ,3 , 80);


