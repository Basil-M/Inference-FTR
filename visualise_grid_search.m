%%%% Script created to visualise log evidence output of grid search

num_points = 250;
old = false;
train = false;
test = true;
% load data

M = csvread('rbf_dump_det_2601.csv');
fetch = @(x) (M(:,x)+M(:,x+3))./sum(M(:,x:x+3),2);
if old
    test_correct = fetch(11);
    train_correct = fetch(15);
elseif size(M,2) == 4
    test_correct = M(:,4);
    train_correct = M(:,3);
else
    test_correct = fetch(13);
    train_correct = fetch(19);
end
%
rbf_var = log10(M(:,1));
prior_var = log10(M(:, 2));
%rbf_var = M(:,1); prior_var = M(:,2);
num_points = sqrt(length(rbf_var));

%add gridfit to path
addpath('gridfitdir');
gx = linspace(min(rbf_var), max(rbf_var), num_points);
gy = linspace(min(prior_var), max(prior_var), num_points);


%% gridfit is a toolbox developed by John D'Errico for smoothing surface data
if test
    g_test=gridfit(rbf_var,prior_var,test_correct,gx,gy,'regularizer','springs','smooth',5,'interp','bilinear');
end
if train
    g_train=gridfit(rbf_var,prior_var,train_correct,gx,gy,'regularizer','springs','smooth',5,'interp','bilinear');
end
figure

colormap(parula(512));
if test
    surf(gx,gy,g_test, 'EdgeAlpha',0.2,'FaceAlpha',1);
end
f_alpha = 1;
if test && train
    hold on;
    f_alpha = 0.5;
end
if train
    surf(gx,gy,g_train,'EdgeAlpha',0.2,'FaceAlpha',f_alpha);
end
xlabel('RBF variance (\sigma_0)')
ylabel('Prior variance (l)')
zlabel('Log Model Evidence')

set(gca,'FontName','Cambria')
limits = [floor(min(rbf_var)) ceil(max(rbf_var))];
k = (limits(2) - limits(1) > 5) + 1;
l_t = unique([limits(1):k:0 mod(limits(2),2):k:limits(2)]);
t = cell(size(l_t));

for l = 1:length(l_t)
    t(l) = {['10^{' num2str(l_t(l)) '}']};
end

xticks(l_t);
yticks(l_t)
set(gca, 'TickLabelInterpreter', 'tex', 'xTickLabels', t, 'yTickLabels', t)


xlim(limits); ylim(limits);% zlim([.4 1])
%set(gca, 'xTickLabels', t, 'yTickLabels', t)
%view(c)