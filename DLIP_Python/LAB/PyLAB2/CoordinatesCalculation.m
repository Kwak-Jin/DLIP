% About  : DLIP Pixel Coordinate Gradient Calculation
% Author : Jin Kwak/21900031
% Created: 24.06.03
clc; clear all; close all;
X = 1;
Y = 2;
Coords =    [1277, 429;
                1, 437;
                2, 325;
             1095, 320;
              197, 435;
              311, 434;
              454, 323;
              545, 323;
              636, 322;
              725, 324;
              418, 433;
              526, 433;
              634, 432;
              743, 431;
              849, 429;
              957, 430;
             1064, 429;
               87, 437];
Upper = [];
Lower = [];
for idx = 1:length(Coords(:,X))
    if Coords(idx,Y) > 400
        Upper = [Upper; Coords(idx,:)];
    else
        Lower = [Lower; Coords(idx,:)];
    end
end
UpperX = round(mean(Upper(:,X)));
UpperY = round(mean(Upper(:,Y)));
LowerX = round(mean(Lower(:,X)));
LowerY = round(mean(Lower(:,Y)));

%%
close all;
plot(Coords(:,X),720-Coords(:,Y),'r*');
xlim([0 1280]);
ylim([0 720]);

%%
Parking_X = 1280 / 13;
Parking_Y = 432 - 323;
