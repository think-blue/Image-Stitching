function [output_image, output_image_overalap, residual, max_inliers, num_matched_points] = ImStich(image1, image2, keypoint_type, sigma, window_size, iterations)
% Usage:
%   Output parameters:
%     output_image:         final Panaromic image (SIFT gives BETTER resuts than harris)
%     image_overlap:        image showing overlap between the two images
%     residual:             residue as defined
%     max_inliers:          Number of inliers from RANSAC 
%     num_matched_points:   Total number of matching pairs 
%          
%   Input parameters:
%       keypoint_type:  string value 'Harris' or 'SIFT'
%       sigma:          standard deviation for harris detector
%       window_size:    radius of neighbourgood for harris detector
%       iterations:     number of iterations for RANSAC

%% converting images to grayscale and double
img_1 = image1;
img_2 = image2;

if(size(img_1, 3) == 3)
    img_1 =  double(rgb2gray(img_1)) ;
else
    img_1 = double(img_1);
end

if(size(img_2, 3) == 3)
    img_2 = double(rgb2gray(img_2)) ;
else
    img_2 = double(img_2);
end


%% Exracting Key points
if strcmpi(keypoint_type, 'SIFT')
    %SIFT keypoints and Descriptors (Done only to compare/ Not required for assignment)
    [f1,d1] = vl_sift(single(img_1));
    [f2,d2] = vl_sift(single(img_2));
    threshold = 1;  % threshold to decide if keypoints match
    threshold_transformation = 0.9;
elseif strcmpi(keypoint_type, 'HARRIS')
    % 2) Extracting Harris Keypoints
    [~, row_1, column_1] = harris(img_1, sigma, 1000, window_size, 0);
    [~, row_2, column_2] = harris(img_2, sigma, 1000, window_size, 0);

    % Describing Harris Keypints using SIFT
    f1 = []; f2 = []; d1 = []; d2 = [];
    
    [fa,da] = vl_sift(single(img_1), 'frames', [row_1'; column_1'; 5*ones( length(row_1),1 )'; zeros( length(row_1),1 )' ], 'orientations');
    [fb,db] = vl_sift(single(img_2), 'frames', [row_2'; column_2'; 5*ones( length(row_2),1 )'; zeros( length(row_2),1 )' ], 'orientations');
    f1 = [f1 fa]; f2 = [f2 fb];
    d1 = [d1 da]; d2 = [d2 db];
    
    
    threshold = 7; % threshold to decide if keypoints match
    threshold_transformation = 8;
end

%% Matching key points


% Matching coorosponding Key points 
normailsed_distance_1 = zscore(double(d1)); %normalising descriptors to z-scores
normalised_distance_2 = zscore(double(d2)); %normalising descriptors to z-scores
distances = pdist2(normailsed_distance_1', normalised_distance_2', 'euclidean');

[row, column] = find(distances < threshold);
pair_1 = f1(1:2, row);
pair_2 = f2(1:2, column);
matched_pairs = [pair_1; pair_2];
num_matched_points = length(matched_pairs);

% Automatic matching
[matches, scores] = vl_ubcmatch(d1, d2) ;
pair_1 = f1(1:2, matches(1,:));
pair_2 = f2(1:2, matches(2,:));
matched_pairs = [pair_1; pair_2];
num_matched_points = length(matched_pairs);

%plotting matching points (Uncomment to plot; HEIGHT OF IMAGE 1 should be equal to HEIGHT OF IMAGE 2)
% figure;
% imshow([image1, image2], [])
% hold on;
% plot(matched_pairs(1,:), matched_pairs(2,:),'rs');
% hold on;
% plot(matched_pairs(3,:) + size(img_1, 2), matched_pairs(4,:),'rs');
% hold on;
% for i = 1:length(matched_pairs)
% line([matched_pairs(1,i)', matched_pairs(3,i)' + size(img_1, 2)], [matched_pairs(2,i)', matched_pairs(4,i)'])
% hold on;
% end

%% Finding Model Parameters using RANSAC
max_inliers = 0; %initialising variable to hold maximum number of inliers
for iteration = 1:iterations
    samples = randsample(length(matched_pairs), 3); %randomoly sampling 3 pairs of points from both images
    A = matched_pairs(1:2, samples)';  % points from first image
    A = [A ones(length(A),1)];  % appending one to account for transalation variables in transormation matrix
    B = matched_pairs(3:4, samples)';  % points from second image
    
    X = A\B;    % calculating Transformation Matrix using 3 pairs of points
    
    % calculating number of inliers
    data_points_A = matched_pairs(1:2,:)'; % all matched keypoints in image 1
    data_points_A = [ data_points_A, ones(length(data_points_A), 1) ];
    transformed_points = data_points_A * X;     % transformed points
    data_points_B = matched_pairs(3:4,:)'; % points in image 2
    error = diag(pdist2(transformed_points, data_points_B));
    good_match = find(error < threshold_transformation);             % threshold = 0.9
    if(length(good_match) > max_inliers)
        max_inliers = length(good_match);
        inliers_index = good_match;
        best_transform = X;
    end
end

%% Using Least squares on the output from RANSAC
A = matched_pairs(1:2,inliers_index)';
A_appended = [A ones(length(A),1)];
B = matched_pairs(3:4,inliers_index)';
X = A_appended\B;

%% calculating Residuals
transformed_points = A_appended * X;
residual = sum(sum((transformed_points - B).^2))/length(B); % residuals as sum of squared errors

%plotting inliers (Uncomment to plot)
% figure;
% subplot(121);imagessc(image1);
% hold on;
% plot(matched_pairs(1,inliers_index), matched_pairs(2,inliers_index),'rs');
% subplot(122);imagesc(image2)
% plot(matched_pairs(3,inliers_index) + size(img_1, 2), matched_pairs(4,inliers_index),'rs');


%% Combining the two images
T1 = [X [0; 0; 1]];
T = maketform('affine',T1);
T.tdata.T;
im2 = image1;
im1 = image2;

[~,xdataim2t,ydataim2t]=imtransform(im2,T);
% now xdataim2t and ydataim2t store the bounds of the transformed im2
xdataout=[min(1,xdataim2t(1)) max(size(im1,2),xdataim2t(2))];
ydataout=[min(1,ydataim2t(1)) max(size(im1,1),ydataim2t(2))];
% let's transform both images with the computed xdata and ydata
im2t=imtransform(im2,T,'XData',xdataout,'YData',ydataout);
im1t=imtransform(im1,maketform('affine',eye(3)),'XData',xdataout,'YData',ydataout);
output_image_overalap=im1t/2+im2t/2;
%figure, imagesc(ims)
imd=uint8(abs(double(im1t)-double(im2t)));
% the casts necessary due to the images' data types
%figure;
%imagesc(imd);
ims=max(im1t,im2t);
output_image = ims;
%figure;
%imagesc(ims);
