%% Global thresholding for nuclear segmentation. Comment this upto k-means method below to use k-means segmentation
im1 = loadImages('Data/Cell 14_YAC128_G/', '*.tif');
im2 = loadImages('Data/Cell 14_YAC128_R/', '*.tif');

fs = 10;
t = (0:length(im1)-1)/fs;

%% calculate z mean
im1_mean = mean(im1,3);
im2_mean = mean(im2,3);

%% segment image by binary thresholding using otsu's method
[BW1,maskedImage1] = segmentImage(im1_mean);
[BW2,maskedImage2] = segmentImage(im2_mean);
% remove mask of image1 from that of image2
BW2(BW1) = 0;
maskedImage2(BW1) = 0;

%% calculate dff
im1_dff = (im1 - im1_mean)./im1_mean;
im2_dff = (im2 - im2_mean)./im2_mean;
%% Apply mask to stack of images
im1_dff = bsxfun(@times, im1_dff, cast(BW1, 'like', im1_dff));
im2_dff = bsxfun(@times, im2_dff, cast(BW2, 'like', im2_dff));

%% get temporal trace of masked images
mean1 = squeeze(mean(im1_dff,[1 2]));
mean2 = squeeze(mean(im2_dff,[1 2]));


% %% k-mean clustering approach for segmentation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% pixel_labels1 = imsegkmeans(uint8(im1_mean),3);
% pixel_labels2 = imsegkmeans(uint8(im2_mean),3);
% 
% %%
% labels = unique(pixel_labels1);
% mean_label_val = [];
% for label = 1:length(labels)
%     label_im = im1_mean;
%     label_im(pixel_labels1~=label)=0;
%     mean_label_val = [mean_label_val mean(nonzeros(label_im),'all')];
% end
% [Y,label1] = max(mean_label_val);
% 
% labels = unique(pixel_labels2);
% mean_label_val = [];
% for label = 1:length(labels)
%     label_im = im2_mean;
%     label_im(pixel_labels2~=label)=0;
%     mean_label_val = [mean_label_val mean(nonzeros(label_im),'all')];
% end
% 
% [Y,label2] = max(mean_label_val);
% %%
% pixel_labels1(pixel_labels1 ~= label1) = 0;
% pixel_labels1(pixel_labels1 ~= 0) = 1;
% pixel_labels2(pixel_labels2 ~= label2) = 0;
% pixel_labels2(pixel_labels2 ~= 0) = 1;
% pixel_labels2(find(pixel_labels1)) = 0;
% %% calculate dff
% im1_dff = (im1 - im1_mean)./im1_mean;
% im2_dff = (im2 - im2_mean)./im2_mean;
% %% Apply mask to stack of images
% im1_dff = bsxfun(@times, im1_dff, cast(pixel_labels1, 'like', im1_dff));
% im2_dff = bsxfun(@times, im2_dff, cast(pixel_labels2, 'like', im2_dff));
% 
% %% get temporal trace of masked images and plot
% mean1 = squeeze(mean(im1_dff,[1 2]));
% mean2 = squeeze(mean(im2_dff,[1 2]));
% figure; 
% subplot(2,2,1); imshowpair(im1_mean, pixel_labels1); title('Nucleus: GCaMP(ROI: Pink)');
% subplot(2,2,2); imshowpair(im2_mean, pixel_labels2); title('Cytosol: RCaMP(ROI: Pink)');
% subplot(2,1,2); plot(mean1, 'g', 'DisplayName','GCaMP'); hold on; plot(mean2, 'r', 'DisplayName','RCaMP'); xlabel('Sample frames'); ylabel('DFF')
% legend;

%% find peaks
[PkAmp1, PkTime1, W1, P1] = findpeaks(mean1,t, 'MinPeakProminence',0.01,'MinPeakDistance',2);
[PkAmp2, PkTime2, W2, P2] = findpeaks(mean2,t, 'MinPeakProminence',0.01,'MinPeakDistance',2);
% findpeaks(mean1,t,'MinPeakProminence',0.01,'MinPeakDistance',0.5,'Annotate','extents')

%% time match the peaks
% we are assuming matching peaks would be max. 2 sec. apart. Anything more
% is not a match
m = ismembertol(PkTime1, PkTime2, 2/max(abs([PkTime1(:);PkTime2(:)])));
PkAmp1 = PkAmp1(m); PkTime1 = PkTime1(m); W1 = W1(m); P1 = P1(m);
PkAmp2 = PkAmp2(m); PkTime2 = PkTime2(m); W2 = W2(m); P2 = P2(m);

%% ratios
RtPkAmp = PkAmp1./PkAmp2; DfPkTime = PkTime1-PkTime2; RtW = W1./W2; RtP = P1./P2;

%% Plot data
figure; 
subplot(2,2,1); imshowpair(im1_mean, maskedImage1); title('Nucleus: GCaMP(ROI: White)');
subplot(2,2,2); imshowpair(maskedImage2, im2_mean); title('Cytosol: RCaMP(ROI: White)');
subplot(2,1,2); plot(t, mean1, 'g', 'DisplayName','GCaMP'); hold on; 
plot(t, mean2, 'r', 'DisplayName','RCaMP'); xlabel('Time (Sec.)'); ylabel('DFF')
plot(PkTime1, PkAmp1, '^g', 'MarkerFaceColor','b', 'DisplayName','GCaMP Peaks'); 
plot(PkTime2, PkAmp2, '^r', 'MarkerFaceColor','b', 'DisplayName','RCaMP Peaks');hold off; grid on
legend;

%% plot ratios and scatter plot
figure;
subplot(2,3,1); scatter(PkTime1, PkTime2); xlabel('Peak time GCaMP'); ylabel('Peak time RCaMP'); title('Peak times'); axis equal;
subplot(2,3,2); scatter(W1, W2); xlabel('Peak widths GCaMP'); ylabel('Peak widths RCaMP'); title('Peak widths');  axis equal;
subplot(2,3,3); scatter(P1, P2); xlabel('Peak prominence GCaMP'); ylabel('Peak prominence RCaMP'); title('Peak prominence');  axis equal;
subplot(2,3,4); plot(DfPkTime); xlabel('Peak #'); ylabel('Difference of peak time(GCaMP-RCaMP)'); title('Peak times');  axis equal;
subplot(2,3,5); plot(RtW); xlabel('Peak #'); ylabel('Ratio peak widths(GCaMP/RCaMP)'); title('Peak widths');
subplot(2,3,6); plot(RtP); xlabel('Peak #'); ylabel('Ratio peak prominence(GCaMP/RCaMP)'); title('Peak prominence');