close all
clear
clc

h = hgload('chebyshev.fig');
%figure('PaperPositionMode', 'auto');
%plot(rand(10));
print(gcf, '-dpdf', '~/adaptive-batch-size/plots/chebyshev.pdf','-fillpage');