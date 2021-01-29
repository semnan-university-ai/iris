clc;
close all;
clear;

% Amir Shokri 9811920009

load fisheriris
f = figure;
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
N = size(meas,1);
title('before normalization');

covarianse_before = cov(meas);
coeff_before = pca(meas);

max = meas(1,1);
min = meas(1,1);

for i = 1 : 150
    for j = 1 : 4
        if(max < meas(i, j))
            max = meas(i, j);
        end
        if(min > meas(i, j))
            min = meas(i, j);
        end
    end
end

for i = 1 : 150
    for j = 1 : 4
        meas(i, j) = ( meas(i, j) - min ) / ( max - min );
    end
end

f = figure;
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
N = size(meas,1);
title('after normalization');

covarianse_after = cov(meas);
coeff_after = pca(meas);