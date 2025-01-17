% Pankaj Gupta <pankajgupta@alumni.ubc.ca>
% University of British Columbia
% Created: April 2020
function Seq = loadImages(imgPath, imgType)
    %imgPath = 'path/to/images/folder/';
    %imgType = '*.png'; % change based on image type
    images  = dir([imgPath imgType]);
    N = length(images);

    % check images
    if( ~exist(imgPath, 'dir') || N<1 )
        display('Directory not found or no matching images found.');
    end

    % preallocate cell
    im = imread([imgPath images(1).name]);
%     Seq(N,size(im,1), size(im,2)) = 0;
    Seq = zeros(size(im,1), size(im,2),N);

    for idx = 1:N
        Seq(:,:,idx) = imread([imgPath images(idx).name]);
    end
end