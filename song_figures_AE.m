%% Init
close all;
clear; clc;

% Set main paths
base_folder = '.\outputs\features\';
path_fig = '.\outputs\figures\'; 
datafolder = '.\outputs\CND\';

%% Parameters
reRefType = 'Mastoids';
cv = 'cvonly';
lags = '-50_350\';
pre = '1-30Hz\';

% for viz of the results
conditions = {'melody', 'song', 'speech'};
p_thresh = 0.05;
average_channels = 1;
title_str = 'Amplitude Envelope';
feature_name = 'A';  
tmin = 0; 
tmax = 300;
corr_factor = 2;

% Viz parameters
chan_of_interest = 47;   
chan_of_interest_str = 'Cz';
channels_of_interest = [37, 47, 30];
channels_of_interest_str = {'Fz', 'Cz', 'Pz'};
t1 = 60;
t2 = 125;
t3 = 200;
ylims_auto = 0;
ylims = [-2, 2];
common_colorbar = 1;

%% Define parameters figures
xlims = [tmin, tmax];
x = 1:3;
conds_for_plot = {'melody', 'sung speech', 'speech'}; 
FontSize = 40;
FontSizeTitle = 55;
set(gca,'DefaultTextFontSize',FontSize)
set(0,'defaultfigurecolor',[1 1 1])
alpha = 0.2;
smth = 1;
sem = 1;

% Define RGB values for blue, green, and red
blue = [0 0.4470 0.7410];
green = [0.4660 0.6740 0.1880];
red = [0.8500 0.3250 0.0980];
colormap = [red; green; blue];

%% Extract results
rcat = [];
gcat = [];
scat = [];
barv = [];
rcat_dec = [];
gcat_dec = [];
scat_dec = [];
barv_dec = [];
rcat_diff = [];
gcat_diff = [];
scat_diff = [];
barv_diff = [];
for idx_cond = 1: length(conditions)
    cond = conditions{idx_cond};
    %% Load EEG data for metadata and channel locs
    eegPreFilename = [datafolder, cond,'\',pre,reRefType,'\pre_dataSub1.mat'];
    load(eegPreFilename,'eeg')

    %% Load TRFs and correlations
    % Build paths
    tfolder = ['outputs/TRFs/' pre, lags, reRefType, '\'];
    prename = [feature_name, '_', cond, '_'];
    
    % Load TRFs
    modelAll_path = [tfolder, prename, 'modelAll_', cv, '.mat'];
    load(modelAll_path, 'modelAll');
    model_all{idx_cond} = modelAll;

    % Load Pearson's encoding
    r_all_path = [tfolder, prename, 'rpredAll_', cv, '.mat'];
    load(r_all_path, 'rpredAll');
    r_all{idx_cond} = rpredAll;

    % Load Pearson's decoding
    tfolder_dec = ['outputs/Decoders/' pre, lags, reRefType, '\'];
    r_all_path = [tfolder_dec, prename, 'rpredAll_', cv, '.mat'];
    load(r_all_path, 'rpredAll');
    r_dec{idx_cond} = rpredAll;

    % Load Pearson's null model  
    prename = [feature_name,'_',cond, '_'];
    r_shu_path = [tfolder, prename, 'rpredShu_' cv, '.mat'];
    r_shu{idx_cond} = load(r_shu_path);
    r_shu{idx_cond} = r_shu{idx_cond}.rpredShu;

    %% Get TRFs for a feature of interest and compute significance
    % Stack normalized models - considering only one feature
    wAvgFeat = [];
    for sub = 1:length(modelAll)
        if isempty(modelAll(sub).w)
            continue
        end
        w = modelAll(sub).w;

        % Select the feature of interest
        w = w(1, :, :);
        
        % normalize
        m = mean(w, 2);
        sd = std(w(:));
        w = w - m;
        w = w/sd; 
        wAvgFeat = [wAvgFeat; w];
    end
    model_avg_feat{idx_cond} = wAvgFeat;

    % ttest on TRF of the feature of interest - variance over subjects
    nlags = size(model_avg_feat{idx_cond}, 2);
    nsbjs = size(model_avg_feat{idx_cond}, 1);
    ttest_vector = [];
    ptest_vector = [];
    if nsbjs > 1
        for lag = 1:nlags
            distr = squeeze(squeeze(model_avg_feat{idx_cond}(:, lag, chan_of_interest)));
            [h,p,ci,stats] = ttest(distr, 0, 'Alpha', 0.05);
            ptest_vector = [ptest_vector; p];
            ttest_vector = [ttest_vector; h];
        end
    end
    ptest_final{idx_cond} = ptest_vector;
    ttest_final{idx_cond} = ttest_vector;

    %% Averaging TRFs and Pearson's across subjects
    % Compute average TRF over subjects
    model_avg_sbj{idx_cond} = mTRFmodelAvg(modelAll,1);

    % Compute average correlation scores across subjects
    rpred_avg_sbj{idx_cond} = squeeze(mean(r_all{idx_cond},1));

    %% Compute statistics on Pearson's for each channel for topographies
    % Get statistics for each channel so to see significant channels
    [tpred{idx_cond}, ppred{idx_cond}] = ttest(r_all{idx_cond}); 

    % Compute average correlation scores (for topographies)
    rdiff{idx_cond} = r_all{idx_cond} - r_shu{idx_cond};
    rdiff_avg_sbj{idx_cond} = squeeze(mean(rdiff{idx_cond},1));

    %% Compute statistics on Pearson's for barplots 
    % (on 1 channel or on the average of all channels)
    if average_channels
        rcat{idx_cond} = squeeze(mean(r_all{idx_cond},2));
    else
        rcat{idx_cond} = r_all{idx_cond}(:, chan_of_interest);
    end 
    [t(idx_cond), p(idx_cond)] = ttest(rcat{idx_cond});
    gcat = [gcat; repmat({conds_for_plot{idx_cond}},length(rcat{idx_cond}),1)];
    scat = [scat; std(rcat{idx_cond})/sqrt(length(rcat{idx_cond}))];
    barv = [barv, mean(rcat{idx_cond})];

    % assign color grey/black according to significnce
    if t(idx_cond) == 0
        C(idx_cond, :) = [.7 .7 .7];
    else
        C(idx_cond, :) = [.0 .0 .0];
    end

    %% Compute statistics on Pearson's DECODERS for barplots 
    % (on 1 channel or on the average of all channels)
    if average_channels
        rcat_dec{idx_cond} = squeeze(mean(r_dec{idx_cond},2));
    else
        rcat_dec{idx_cond} = r_dec{idx_cond};
    end 
    [t_dec(idx_cond), p_dec(idx_cond)] = ttest(rcat_dec{idx_cond});
    gcat_dec = [gcat_dec; repmat({conds_for_plot{idx_cond}},length(rcat_dec{idx_cond}),1)];
    scat_dec = [scat_dec; std(rcat_dec{idx_cond})/sqrt(length(rcat_dec{idx_cond}))];
    barv_dec = [barv_dec, mean(rcat_dec{idx_cond})];

    % assign color grey/black according to significnce
    if t_dec(idx_cond) == 0
        C_dec(idx_cond, :) = [.7 .7 .7];
    else
        C_dec(idx_cond, :) = [.0 .0 .0];
    end

    %% Compute statistics on Person's differences with shuffeling
    % (on 1 channel or on the average of all channels)
    if average_channels
        rcat_diff{idx_cond} = mean(r_all{idx_cond},2) - mean(r_shu{idx_cond},2);
    else
        rcat_diff{idx_cond} = r_all{idx_cond}(:, chan_of_interest) - r_shu{idx_cond}(:, chan_of_interest);
    end

    % Get statistics for barplots
    [t_diff(idx_cond), p_diff(idx_cond)] = ttest(rcat_diff{idx_cond});
    gcat_diff = [gcat_diff; repmat({cond},length(rcat_diff{idx_cond}),1)];
    scat_diff = [scat_diff; std(rcat_diff{idx_cond})/sqrt(length(rcat_diff{idx_cond}))];
    barv_diff = [barv_diff, mean(rcat_diff{idx_cond})];  

    % assign color grey/black according to significnce
    if t_diff(idx_cond) == 0
        C_diff(idx_cond, :) = [.7 .7 .7];
    else
        C_diff(idx_cond, :) = [.0 .0 .0];
    end

end

%% Plots topographies
fig = figure(); 
% Each column is a condition
n_conditions = length(conditions);
n_rows = 1;

for idx_cond = 1:n_conditions
    %% Person's for each channel on topographies
    subplot(n_rows,n_conditions,idx_cond) 
    title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle)

    % Compute significance
    rplot = rpred_avg_sbj{idx_cond};
    [tpred{idx_cond}, pvalues] = ttest(r_all{idx_cond}); 
%     pvalues = pvalues.';
%     pvalues = -log10(pvalues);
%     pvalues = pvalues / max(pvalues);
%     [~,qvalues] = mafdr(pvalues);
%     qvalues = qvalues/ max(qvalues);
%     pvalues = 10.^(-qvalues);
    rplot(pvalues>p_thresh) = 0;

    % Get zlim colorbar
    if common_colorbar
        zlim_max = max(cell2mat(rpred_avg_sbj));
    else
        zlim_max = max(rplot);
    end

    % Plot topographies
    topoplot(rplot, eeg.chanlocs,'maplimits',[0, zlim_max],'whitebk','on')
    if idx_cond == 1
        c = colorbar;
        c.Location = 'westoutside';
        c.Label.String = ['r'];   
        c.FontSize = FontSize;
    end

    set(gcf,'color','w');

end

figure('Renderer', 'painters', 'Position', [10 10 500 1400]) % 1200 1000]
% Each column is a condition
n_conditions = length(conditions);
n_rows = 3;

for idx_cond = 1:n_conditions
    %% Person's for each channel on topographies
    subplot(n_rows,1,idx_cond) 
    title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle)

    % Compute significance
    rplot = rpred_avg_sbj{idx_cond};
    [tpred{idx_cond}, pvalues] = ttest(r_all{idx_cond});
    pvalues = pvalues.';
    pvalues = pvalues*64;
    rplot(pvalues>p_thresh) = 0;

    % Get zlim colorbar
    if common_colorbar
        zlim_max = max(cell2mat(rpred_avg_sbj));
    else
        zlim_max = max(rplot);
    end

    % Plot topographies
    topoplot(rplot, eeg.chanlocs,'maplimits',[0, zlim_max],'whitebk','on')
    if idx_cond == 1
        c = colorbar;
        c.Location = 'westoutside';
        c.Label.String = ['r'];   
        c.FontSize = FontSize;
    end

    set(gcf,'color','w');

end


%% Plots topographies of differences with shuff
fig = figure(); 
% Each column is a condition
n_conditions = length(conditions);
n_rows = 1;

for idx_cond = 1:n_conditions
    %% Person's for each channel on topographies
    subplot(n_rows,n_conditions,idx_cond) 
    title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle)

    % Compute significance
    rplot = rdiff_avg_sbj{idx_cond};
    [tpred{idx_cond}, ppred{idx_cond}] = ttest(rdiff{idx_cond}); 
%     ppred{idx_cond} = mafdr(ppred{idx_cond});
    rplot(ppred{idx_cond}>p_thresh) = 0;

    % Get zlim colorbar
    if common_colorbar
        zlim_max = max(cell2mat(rpred_avg_sbj));
    else
        zlim_max = max(rplot);
    end

    % Plot topographies
    topoplot(rplot, eeg.chanlocs,'maplimits',[0, zlim_max],'whitebk','on')
    if idx_cond == 1
        c = colorbar;
        c.Location = 'westoutside';
        c.Label.String = ['Pearson''s correlation'];   
        c.FontSize = FontSize;
    end

    set(gcf,'color','w');

end

%% Plots topographies and TRFs
fig = figure(); 
% Each column is a condition
n_conditions = length(conditions);
n_rows = 2;

for idx_cond = 1:n_conditions
    %% Person's for each channel on topographies
    subplot(n_rows,n_conditions,idx_cond) 
    title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle)

    % Compute significance
    rplot = rpred_avg_sbj{idx_cond};
    [tpred{idx_cond}, ppred{idx_cond}] = ttest(r_all{idx_cond}); 
%     ppred{idx_cond} = mafdr(ppred{idx_cond});
    rplot(ppred{idx_cond}>p_thresh) = 0;

    % Get zlim colorbar
    if common_colorbar
        zlim_max = max(cell2mat(rpred_avg_sbj));
    else
        zlim_max = max(rplot);
    end

    % Plot topographies
    topoplot(rplot, eeg.chanlocs,'maplimits',[0, zlim_max],'whitebk','on')
    if idx_cond == 1
        c = colorbar;
        c.Location = 'westoutside';
        c.Label.String = ['Pearson''s correlation'];   
        c.FontSize = FontSize;
    end

    %% Plot TRFs for a given feature and channel with significance
    subplot(n_rows,n_conditions,idx_cond+n_conditions) 
    avgModel = model_avg_sbj{idx_cond};
    ax = plot(avgModel.t,squeeze(avgModel.w(1,:,:)));
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5)  
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    xlabel('Time lag (ms)', 'FontSize', FontSize)
    if idx_cond == 1
        ylabel('TRF amplitude (a.u.)', 'FontSize', FontSize)
    end
    xlim(xlims)
    if ylims_auto
        ylim auto
    else
        ylim(ylims) 
    end
    hold on
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',FontSize)
    set(gca,'XTickLabelMode','auto')
    a = get(gca,'YTickLabel');  
    set(gca,'YTickLabel',a,'fontsize',FontSize)
    set(gca,'YTickLabelMode','auto')
    set(gcf,'color','w');

%     % ttest on GFP
%     ttest_gfp = [];
%     if size(avgModel.w, 3) > 1
%         for lag = 1:size(avgModel.w, 2)
%             distr = squeeze(squeeze(avgModel.w(1, lag, :)));
%             [h,p,ci,stats] = ttest(distr, 0, 'Alpha', 0.05);
%             ttest_gfp = [ttest_gfp; h];
%         end
%     end
%     shaded_patch_significant_timepoints(avgModel.t, ttest_gfp, ax)


%     ax = stdshade(squeeze(model_avg_feat{idx_cond}(:, :, chan_of_interest)), ...
%                   alpha,colormap(idx_cond, :),model_avg_sbj{idx_cond}.t,smth,sem);
%     if ylims_auto
%         ylim auto
%     else
%         ylim(ylims) 
%     end
%     hold on
%     shaded_patch_significant_timepoints(model_avg_sbj{idx_cond}.t, ttest_final{idx_cond}, ax)
%     yline(0, '-', 'Alpha', 0.5)
%     xline(0, '-', 'Alpha', 0.5) 
%     xline(t1, '--', 'Alpha', 0.9) 
%     xline(t2, '--', 'Alpha', 0.9) 
%     xline(t3, '--', 'Alpha', 0.9) 
%     xlabel('Time lag (ms)', 'FontSize', FontSize) 
%     ylabel(['TRF at ', chan_of_interest_str], 'FontSize', FontSize)
%     xlim(xlims)
%     axis square
%     a = get(gca,'XTickLabel');  
%     set(gca,'XTickLabel',a,'fontsize',FontSize)
%     set(gca,'XTickLabelMode','auto')
%     a = get(gca,'YTickLabel');  
%     set(gca,'YTickLabel',a,'fontsize',FontSize)
%     set(gca,'YTickLabelMode','auto')
%     set(gcf,'color','w');

end

%% Plot barplots with statistics
figure('Renderer', 'painters', 'Position', [10 10 2000 1400]) % 1200 1000]

% Plot barplot Pearson's - encoding
P = nan(numel(barv), numel(barv));
% P(1,2) = anova1([cell2mat(rcat(1)),cell2mat(rcat(2))],[],'off')*corr_factor;
% P(1,3) = anova1([cell2mat(rcat(1)),cell2mat(rcat(3))],[],'off')*corr_factor;
% P(2,3) = anova1([cell2mat(rcat(2)),cell2mat(rcat(3))],[],'off')*corr_factor;
PT = P'; 
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);

subplot(1,2,1)
superbar(barv, 'E', scat', 'P', P, 'BarFaceColor', C, ...
    'Orientation', 'v', 'ErrorbarStyle', 'T', 'PLineOffset', 0.005, ...
    'PStarFontSize', 24);
title('Encoding', 'FontSize', FontSizeTitle);
set(gca,'xtick',x,'xticklabel',{'melody', 'sung s.', 'speech'}, 'FontSize', FontSize)
xtickangle(20)
% xlabel('Condition', 'FontSize', FontSize)
ylabel('r', 'FontSize', FontSize) 
ylim auto
a = get(gca,'YTickLabel');  
set(gca,'YTickLabel',a,'fontsize',FontSize)
set(gca,'YTickLabelMode','auto')
set(gcf,'color','w');
% legend({'p<0.001'})
% legend('boxoff')

% Compute cross-significance for Pearson's difference
P = nan(numel(barv_dec), numel(barv_dec));
%     P(1,2) = anova1([cell2mat(rcat_dec(1)),cell2mat(rcat_dec(2))],[],'off')*corr_factor;
%     P(1,3) = anova1([cell2mat(rcat_dec(1)),cell2mat(rcat_dec(3))],[],'off')*corr_factor;
%     P(2,3) = anova1([cell2mat(rcat_dec(2)),cell2mat(rcat_dec(3))],[],'off')*corr_factor;
PT = P'; 
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);

% Plot barplot Pearson's - decoding
subplot(1,2,2)
superbar(barv_dec, 'E', scat_dec', 'P', P, 'BarFaceColor', C_dec, ...
    'Orientation', 'v', 'ErrorbarStyle', 'T', 'PLineOffset', 0.009, ...
    'PStarFontSize', 24);    
title('Decoding', 'FontSize', FontSizeTitle);
set(gca,'xtick',x,'xticklabel',{'melody', 'sung s.', 'speech'}, 'FontSize', FontSize)
%     xlabel('Condition', 'FontSize', FontSize)
xtickangle(20)
%     ylabel('Pearson correlation (r)', 'FontSize', FontSize) 
ylim auto
a = get(gca,'YTickLabel');  
set(gca,'YTickLabel',a,'fontsize',FontSize)
set(gca,'YTickLabelMode','auto')
set(gcf,'color','w');
%     legend({'p<0.001'}) 
%     legend('boxoff')
% print(fig, [path_fig, feature_name, '_r_box_'],'-r800','-dpng');

% %% Plot barplots with statistics
% fig = figure(); 
% n_rows = 1;
% for idx_cond = 1:n_conditions
%     % (on 1 channel or on the average of all channels)
%     if average_channels
%         r_dist = squeeze(mean(r_all{idx_cond},2));
%         r_null = squeeze(mean(r_shu{idx_cond},2));
%     else
%         r_dist = r_all{idx_cond}(:, chan_of_interest);
%         r_null = r_shu{idx_cond}(:, chan_of_interest);
%     end 
% 
%     m_dist = mean(r_dist);
%     m_null = mean(r_null);
%     sd_dist = std(r_dist/sqrt(length(r_dist)));
%     sd_null = std(r_null/sqrt(length(r_null)));
%     [t_dist, p_dist] = ttest(r_dist);
%     [t_null, p_null] = ttest(r_null);
% 
%     gcat = ['data'; 'null'];
%     scat = [sd_dist, sd_null];
%     barv = [m_dist, m_null];
% 
%     % assign color grey/black according to significnce
%     C(1, :) = [.0 .0 .0];
%     if t_dist == 0
%         C(1, :) = [.7 .7 .7];
%     end
%     C(2, :) = [.0 .0 .0];
%     if t_null == 0
%         C(2, :) = [.7 .7 .7];
%     end
% 
%     % Compute cross-significance for Pearson's
%     P = nan(numel(barv), numel(barv));
%     P(1,2) = anova1([r_dist,r_null],[],'off');
%     PT = P'; 
%     lidx = tril(true(size(P)), -1);
%     P(lidx) = PT(lidx);
% 
%     % Plot barplot Pearson's
%     
%     ax = subplot(n_rows,n_conditions,idx_cond);
%     superbar(barv, 'E', scat', 'P', P, 'BarFaceColor', C, ...
%         'Orientation', 'v', 'ErrorbarStyle', 'T', 'PLineOffset', 0.005);
%     title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle);
%     set(gca, 'xtick',x,'xticklabel',gcat, 'FontSize', FontSize)
%     ylim([-0.01, 0.06])
% 
%     if idx_cond == 1
%         ylabel('Pearson corr (r)', 'FontSize', FontSize)
%     else
%         ax = gca;
%         ax.YAxis.Visible = 'off';   
%     end
%     if idx_cond == 3
%         legend({'p<0.05', 'p>0.05'})
%         legend('boxoff')
% %         legend('Location','northeastoutside')
%     end
% end

% %% Plot barplots with statistics
% fig = figure(); 
% n_rows = 1;
% 
% gcat = [];
% scat = [];
% barv = [];
% C = [];
% % P = nan(numel(2*n_conditions), numel(2*n_conditions));
% for idx_cond = 1:n_conditions
%     % (on 1 channel or on the average of all channels)
%     if average_channels
%         r_dist = squeeze(mean(r_all{idx_cond},2));
%         r_null = squeeze(mean(r_shu{idx_cond},2));
%     else
%         r_dist = r_all{idx_cond}(:, chan_of_interest);
%         r_null = r_shu{idx_cond}(:, chan_of_interest);
%     end 
% 
%     m_dist = mean(r_dist);
%     m_null = mean(r_null);
%     sd_dist = std(r_dist/sqrt(length(r_dist)));
%     sd_null = std(r_null/sqrt(length(r_null)));
%     [t_dist, p_dist] = ttest(r_dist);
%     [t_null, p_null] = ttest(r_null);
% 
%     gcat = [gcat; 'data'; 'null'];
%     scat = [scat; sd_dist, sd_null];
%     barv = [barv; m_dist, m_null];
% 
%     % assign color grey/black according to significnce
%     C_dist = [.0 .0 .0];
%     if t_dist == 0
%         C_dist = [.7 .7 .7];
%     end
%     C_null = [.0 .0 .0];
%     if t_null == 0
%         C_null = [.7 .7 .7];
%     end
% 
%     C = [C; C_dist; C_null];
% 
%     P(2*idx_cond-1,2*idx_cond) = anova1([r_dist,r_null],[],'off');
% end
% 
% % Compute cross-significance for Pearson's;
% PT = P'; 
% lidx = tril(true(size(P)), -1);
% P(lidx) = PT(lidx);
% 
% % Plot barplot Pearson's
% superbar(barv, 'E', scat', 'P', P, 'BarFaceColor', C, ...
%     'Orientation', 'v', 'ErrorbarStyle', 'T', 'PLineOffset', 0.005);
% %title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle);
% set(gca, 'xtick',x,'xticklabel',gcat, 'FontSize', FontSize)
% ylim([-0.01, 0.06])
% ylabel('Pearson corr (r)', 'FontSize', FontSize)
% legend({'p<0.05', 'p>0.05'})
% legend('boxoff')
% 
% 
% 

%% Plot TRFs and weights on topographies
fig = figure();  
n_rows = 4;
for idx_cond = 1:n_conditions
    subplot(n_rows,n_conditions,idx_cond) 
    avgModel = model_avg_sbj{idx_cond};
    ax = plot(avgModel.t,squeeze(avgModel.w(1,:,:)));
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5)  
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    xlabel('Time lag (ms)', 'FontSize', FontSize)
    if idx_cond == 1
        ylabel('TRF amplitude (a.u.)', 'FontSize', FontSize)
    end
    xlim(xlims)
    if ylims_auto
        ylim auto
    else
        ylim(ylims) 
    end
    hold on
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',FontSize)
    set(gca,'XTickLabelMode','auto')
    a = get(gca,'YTickLabel');  
    set(gca,'YTickLabel',a,'fontsize',FontSize)
    set(gca,'YTickLabelMode','auto')
    set(gcf,'color','w');
 
    % xlim
    [topo_minValue,idx_min] = min(abs(avgModel.t-tmin));   
    [topo_maxValue,idx_max] = min(abs(avgModel.t-tmax));

    % ylim
    ylim_max = max(max(abs(avgModel.w(:,idx_min:idx_max,:)),[],3),[],2);
    ylim_min = min(min(avgModel.w(:,idx_min:idx_max,:),[],3),[],2);
    
    % zlim
    lim = max(max(abs(avgModel.w(1,idx_min:idx_max,:)),[],3),[],2);
    
    % topo time values
    [topo1_val, topo1_idx] = min(abs(avgModel.t - t1));
    [topo2_val, topo2_idx] = min(abs(avgModel.t - t2));
    [topo3_val, topo3_idx] = min(abs(avgModel.t - t3));
    
    % Plot avg TRF model
    subplot(n_rows,n_conditions,idx_cond+n_conditions)
    topoplot(avgModel.w(1,topo1_idx,:),eeg.chanlocs,'maplimits',[-lim,lim],'whitebk','on')
    title([num2str(avgModel.t(topo1_idx)),' ms'], 'FontSize', FontSize)

    subplot(n_rows,n_conditions,idx_cond+(2*n_conditions))
    topoplot(avgModel.w(1,topo2_idx,:),eeg.chanlocs,'maplimits',[-lim,lim],'whitebk','on')
    title([num2str(avgModel.t(topo2_idx)),' ms'], 'FontSize', FontSize)
    
    subplot(n_rows,n_conditions,idx_cond+(3*n_conditions))
    topoplot(avgModel.w(1,topo3_idx,:),eeg.chanlocs,'maplimits',[-lim,lim],'whitebk','on')
    title([num2str(avgModel.t(topo3_idx)),' ms'], 'FontSize', FontSize)
end


%% Plot TRFs and weights on topographie - one plot for each condition
n_rows = 3;
for idx_cond = 1:n_conditions
    figure('Renderer', 'painters', 'Position', [10 10 1000 1400])
    subplot(n_rows,3,[1:6]) 
    avgModel = model_avg_sbj{idx_cond};
    ax = plot(avgModel.t,squeeze(avgModel.w(1,:,:)));
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5)  
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    title(conds_for_plot{idx_cond}, 'FontSize', FontSizeTitle)
    xlabel('time lag (ms)', 'FontSize', FontSize)
    if idx_cond == 1
        ylabel('TRF amplitude (a.u.)', 'FontSize', FontSize)
    end
    xlim(xlims)
    if ylims_auto
        ylim auto
    else
        ylim(ylims) 
    end
    hold on
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',FontSize)
    set(gca,'XTickLabelMode','auto')
    a = get(gca,'YTickLabel');  
    set(gca,'YTickLabel',a,'fontsize',FontSize)
    set(gca,'YTickLabelMode','auto')
    set(gcf,'color','w');
 
    % xlim
    [topo_minValue,idx_min] = min(abs(avgModel.t-tmin));   
    [topo_maxValue,idx_max] = min(abs(avgModel.t-tmax));

    % ylim
    ylim_max = max(max(abs(avgModel.w(:,idx_min:idx_max,:)),[],3),[],2);
    ylim_min = min(min(avgModel.w(:,idx_min:idx_max,:),[],3),[],2);
    
    % zlim
    lim = max(max(abs(avgModel.w(1,idx_min:idx_max,:)),[],3),[],2);
    
    % topo time values
    [topo1_val, topo1_idx] = min(abs(avgModel.t - t1));
    [topo2_val, topo2_idx] = min(abs(avgModel.t - t2));
    [topo3_val, topo3_idx] = min(abs(avgModel.t - t3));
    
    % Plot avg TRF model
    subplot(n_rows,3,7)
    topoplot(avgModel.w(1,topo1_idx,:),eeg.chanlocs,'maplimits',[-lim,lim],'whitebk','on')
    title([num2str(t1),' ms'], 'FontSize', FontSize, 'Units', 'normalized', 'Position', [0.5, -0.25, 0]);
    if idx_cond == 1
        c = colorbar;
        c.Location = 'westoutside';
        c.Label.String = ['weights'];   
        c.FontSize = FontSize;
    end

    subplot(n_rows,3,8)
    topoplot(avgModel.w(1,topo2_idx,:),eeg.chanlocs,'maplimits',[-lim,lim],'whitebk','on')
    title([num2str(t2),' ms'], 'FontSize', FontSize, 'Units', 'normalized', 'Position', [0.5, -0.25, 0]);
    
    subplot(n_rows,3,9)
    topoplot(avgModel.w(1,topo3_idx,:),eeg.chanlocs,'maplimits',[-lim,lim],'whitebk','on')
    title([num2str(t3),' ms'], 'FontSize', FontSize, 'Units', 'normalized', 'Position', [0.5, -0.25, 0]);

    set(gcf,'color','w');
end

%% Plot butterfly plots
fig = figure();  %'Renderer', 'painters', 'Position', [10 10 350 800]);
sgtitle(title_str)

n_rows = 2;
for idx_cond = 1:n_conditions
    subplot(n_rows,n_conditions,idx_cond) 
    avgModel = model_avg_sbj{idx_cond};
    plot(avgModel.t,squeeze(avgModel.w(1,:,:)))
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5)  
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    title(['TRF ', conds_for_plot{idx_cond}], 'FontSize', FontSize)
    xlabel('Time lag (ms)', 'FontSize', FontSize)
    ylabel('amplitude (a.u.)', 'FontSize', FontSize)
    xlim(xlims)
    ylim auto
%     axis square

    [topo_minValue,idx_min] = min(abs(avgModel.t-tmin));  
    [topo_minValue,idx_max] = min(abs(avgModel.t-0));   
    meanGFP = mean(std(avgModel.w(1,idx_min:idx_max,:),[],3));
    subplot(n_rows,n_conditions,idx_cond+n_conditions) 
    plot(avgModel.t,std(avgModel.w(1,:,:),[],3))
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5)  
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    yline(2*meanGFP, '--', 'Alpha', 0.9) 
    title(['GFP ', conds_for_plot{idx_cond}], 'FontSize', FontSize)
    xlabel('Time lag (ms)', 'FontSize', FontSize)
    ylabel('amplitude (a.u.)', 'FontSize', FontSize)
    xlim(xlims)
    ylim auto
%     axis square
end

%% Plot TRF of channels of interest
fig = figure();
tiledlayout(1, length(channels_of_interest), 'TileSpacing','tight');

n_rows = 1;
for idx_ch = 1:length(channels_of_interest)
    nexttile
    for idx_cond = 1:n_conditions    
        stdshade(squeeze(model_avg_feat{idx_cond}(:, :, channels_of_interest(idx_ch))), ...
                 alpha,colormap(idx_cond, :),model_avg_sbj{idx_cond}.t,smth,sem);
        hold on
    end
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5) 
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    if idx_ch == 3
        legend({'melody', '', 'sung speech', '', 'speech', '', ''}, 'FontSize', FontSize)
    end
    title(channels_of_interest_str(idx_ch), 'FontSize', FontSize)
    xlabel('Time lag (ms)', 'FontSize', FontSize) 
    if idx_ch == 1
        ylabel('TRF amplitude (a.u.)', 'FontSize', FontSize)
    end
    xlim(xlims)
    ylim(ylims)
    axis square
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',FontSize)
    set(gca,'XTickLabelMode','auto')
    a = get(gca,'YTickLabel');  
    set(gca,'YTickLabel',a,'fontsize',FontSize)
    set(gca,'YTickLabelMode','auto')
    set(gcf, 'Color', 'none');    

end

%% Plot TRF of channels of interest with Cohen's d
n_rows = 3;
for idx_ch = 1:length(channels_of_interest)
    fig = figure('Renderer', 'painters', 'Position', [10 10 1000 1400]);
    subplot(n_rows,1,[1:2]) 
    for idx_cond = 1:n_conditions    
        stdshade(squeeze(model_avg_feat{idx_cond}(:, :, channels_of_interest(idx_ch))), ...
                 alpha,colormap(idx_cond, :),model_avg_sbj{idx_cond}.t,smth,sem);
        hold on
    end
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5) 
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 
    if idx_ch == 3
        legend({'melody', '', 'sung speech', '', 'speech', '', ''}, 'FontSize', FontSize)
    end
    title(channels_of_interest_str(idx_ch), 'FontSize', FontSize)
    if idx_ch == 1
        ylabel('TRF amplitude (a.u.)', 'FontSize', FontSize)
        yticks([-2 -1 0 1 2]);
        a = get(gca,'YTickLabel');  
        set(gca,'YTickLabel',a,'fontsize',FontSize)
    else
        yticklabels({});
    end
    xticklabels({});
    xlim(xlims)
    ylim(ylims) 

    % Compute effect size
    d_song_melody = [];
    d_song_speech = [];
    if size(model_avg_feat{1}, 3) > 1
        for lag = 1:size(model_avg_feat{1}, 2)
    
            distr_melody = squeeze(model_avg_feat{1}(:, lag, channels_of_interest(idx_ch)));
            distr_song   = squeeze(model_avg_feat{2}(:, lag, channels_of_interest(idx_ch)));
            distr_speech = squeeze(model_avg_feat{3}(:, lag, channels_of_interest(idx_ch)));
    
            [h, p, ci, stats] = ttest2(distr_song, distr_melody, 'vartype', 'unequal');
            d_song_melody = [d_song_melody; abs(stats.tstat / sqrt(stats.df + 1))];
    
            [h, p, ci, stats] = ttest2(distr_song, distr_speech, 'vartype', 'unequal');
            d_song_speech = [d_song_speech; abs(stats.tstat / sqrt(stats.df + 1))];
        end
    end

    subplot(n_rows,1,3) 
    plot(avgModel.t,d_song_melody, 'color', red, 'LineWidth', 2);
    hold on
    plot(avgModel.t,d_song_speech, 'color', blue, 'LineWidth', 2);
    xlabel('Time lag (ms)', 'FontSize', FontSize)
    if idx_ch == 1
        ylabel('Effect size', 'FontSize', FontSize)
        yticks([0.2 0.5 0.8]);
        a = get(gca,'YTickLabel');  
        set(gca,'YTickLabel',a,'fontsize',FontSize)
    else
        yticklabels({});
    end
    xlim(xlims)
    ylim([0, 1])
    hold on
    yline(0, '-', 'Alpha', 0.5)
    xline(0, '-', 'Alpha', 0.5) 
    xline(t1, '--', 'Alpha', 0.9) 
    xline(t2, '--', 'Alpha', 0.9) 
    xline(t3, '--', 'Alpha', 0.9) 

    yline(0.2, '-', 'LineWidth', 2)  % 'Small effect size'
    yline(0.5, '-', 'LineWidth', 2)  % 'Medium effect size'
    yline(0.8, '-', 'LineWidth', 2)  % 'Large effect size'

    if idx_ch == 3
        legend({'song-melody', 'song-speech'}, 'FontSize', FontSize)
    end

    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',FontSize)
    set(gca,'XTickLabelMode','auto')
    set(gcf,'color','w');
end

%% Topografia
figure;
topoplot(ones(64,1),eeg.chanlocs,'electrodes','numbers')

%% Plots difference topographies - correlations
fig = figure(); 

rplot_melody = rpred_avg_sbj{1};
rplot_song   = rpred_avg_sbj{2};
rplot_speech = rpred_avg_sbj{3};
rdiff_song_melody = rplot_song - rplot_melody;
rdiff_song_speech = rplot_song - rplot_speech;

rall_melody = r_all{1};
rall_song   = r_all{2};
rall_speech = r_all{3};
[tpred_song_melody, ppred_song_melody] = ttest(rall_song-rall_melody); 
[tpred_song_speech, ppred_song_speech] = ttest(rall_song-rall_speech); 

ppred_song_melody = ppred_song_melody*64;
ppred_song_speech = ppred_song_speech*64;
% [~,ppred_song_melody] = mafdr(ppred_song_melody);
% [~,ppred_song_speech] = mafdr(ppred_song_speech);

rdiff_song_melody(ppred_song_melody>p_thresh) = 0;
rdiff_song_speech(ppred_song_speech>p_thresh) = 0;

r_diff{1} = rdiff_song_melody;
r_diff{2} = rdiff_song_speech;

% Get zlim colorbar
zlim_max = max(max(cell2mat(r_diff)));
zlim_min = min(min(cell2mat(r_diff)));

subplot(1,2,1) 
title('Song-melody', 'FontSize', FontSizeTitle)
rplot = rdiff_song_melody;
if common_colorbar
    if zlim_max<=zlim_min
        zlim_min = 0;
        zlim_max = 1;
    end
    lims = [zlim_min, zlim_max];
else
    zlim_min = min(rplot);
    zlim_max = max(rplot);
    if zlim_max<=zlim_min
        zlim_min = 0;
        zlim_max = 1;
    end
    lims = [zlim_min, zlim_max];
end
topoplot(rplot, eeg.chanlocs,'maplimits',lims,'whitebk','on')
c = colorbar;
c.Location = 'westoutside';
c.Label.String = ['Pearson correlation (r)'];   
c.FontSize = FontSize;

subplot(1,2,2) 
title('Song-speech', 'FontSize', FontSizeTitle)
rplot = rdiff_song_speech;
if common_colorbar
    if zlim_max<=zlim_min
        zlim_min = 0;
        zlim_max = 1;
    end
    lims = [zlim_min, zlim_max];
else
    zlim_min = min(rplot);
    zlim_max = max(rplot);
    if zlim_max<=zlim_min
        zlim_min = 0;
        zlim_max = 1;
    end
    lims = [zlim_min, zlim_max];
end
topoplot(rplot, eeg.chanlocs,'maplimits',lims,'whitebk','on')
c = colorbar;
set(c, 'Orientation', 'horizontal');
c.Location = 'westoutside';
c.Label.String = ['Pearson correlation (r)'];   
c.FontSize = FontSize;

set(gcf,'color','w');


%% Plots difference topographies - TRF weights
figure('Renderer', 'painters', 'Position', [10 10 1000 1400]) % 1200 1000]

[topo1_val, topo1_idx] = min(abs(avgModel.t - t1));
[topo2_val, topo2_idx] = min(abs(avgModel.t - t2));
[topo3_val, topo3_idx] = min(abs(avgModel.t - t3));
topo_idxs = [topo1_idx, topo2_idx, topo3_idx];
time_values = [t1, t2, t3];

TRF_melody = model_avg_sbj{1}.w;
TRF_song   = model_avg_sbj{2}.w;
TRF_speech = model_avg_sbj{3}.w;
TRFdiff_song_melody = TRF_song - TRF_melody;
TRFdiff_song_speech = TRF_song - TRF_speech;


i = 1;
for lag = topo_idxs
    p_song_melody = [];
    p_song_speech = [];
    for ch = 1:size(model_avg_feat{1}, 3)
        distr_melody = squeeze(model_avg_feat{1}(:, lag, ch));
        distr_song   = squeeze(model_avg_feat{2}(:, lag, ch));
        distr_speech = squeeze(model_avg_feat{3}(:, lag, ch));

        [h, p, ci, stats] = ttest2(distr_song, distr_melody, 'Alpha', 0.05);
        p_song_melody = [p_song_melody; p];

        [h, p, ci, stats] = ttest2(distr_song, distr_speech, 'Alpha', 0.05);
        p_song_speech = [p_song_speech; p];
    end
    [~, p_s_m{i}] = mafdr(p_song_melody);
    [~, p_s_s{i}] = mafdr(p_song_speech);
%     p_s_m{i} = p_song_melody*64;
%     p_s_s{i} = p_song_speech*64;
    i = i + 1;
end

for idx_lag = 1:3
    subplot(3,2,2*idx_lag-1)


    % Add text just next to each row of subplots
    row1pos = get(subplot(3,2,2*idx_lag-1), 'Position');
    xpos = row1pos(1);
    ypos1 = row1pos(2);
    text(xpos, ypos1, [num2str(time_values(idx_lag)),' ms'], 'FontSize', FontSize);

    TRFtoviz = abs(TRFdiff_song_melody(1,topo_idxs(idx_lag),:));
    TRFtoviz(p_s_m{idx_lag}>p_thresh) = 0;
    if common_colorbar
        TRF_diff{1} = TRFdiff_song_melody(1,topo_idxs,:);
        TRF_diff{2} = TRFdiff_song_speech(1,topo_idxs,:);
        zlim_max = max(max(abs(cell2mat(TRF_diff))));
        zlim_min = min(min(abs(cell2mat(TRF_diff))));
    else
        zlim_max = max(abs(TRFtoviz));
        zlim_min = min(abs(TRFtoviz));
    end
    if zlim_max<=zlim_min
        zlim_min = 0;
        zlim_max = 1;
    end
    topoplot(TRFtoviz,eeg.chanlocs,'maplimits',[zlim_min,zlim_max],'whitebk','on')
    if idx_lag == 1
        title('S - M', 'FontSize', FontSizeTitle)
        c = colorbar;
        c.Orientation = 'horizontal';
        c.Location = 'westoutside';
        c.Label.String = ['weights'];   
        c.FontSize = FontSize;
    end
%     c = colorbar;
%     c.Orientation = 'horizontal';
%     c.Location = 'westoutside';
%     c.Label.String = ['weights'];   
%     c.FontSize = FontSize;
%     c.Ticks = [round(zlim_min,2), round(zlim_max,2)];

  
    % Calculate FDR adjusted p-values using the mafdr function
    subplot(3,2,2*idx_lag) 
    TRFtoviz = abs(TRFdiff_song_speech(1,topo_idxs(idx_lag),:));
    TRFtoviz(p_s_s{idx_lag}>p_thresh) = 0;
    if common_colorbar
        TRF_diff{1} = TRFdiff_song_melody(1,topo_idxs,:);
        TRF_diff{2} = TRFdiff_song_speech(1,topo_idxs,:);
        zlim_max = max(max(abs(cell2mat(TRF_diff))));
        zlim_min = min(min(abs(cell2mat(TRF_diff))));
    else
        zlim_max = max(abs(TRFtoviz));
        zlim_min = min(abs(TRFtoviz));
    end
    if zlim_max<=zlim_min
        zlim_min = 0;
        zlim_max = 1;

    end
    topoplot(TRFtoviz,eeg.chanlocs,'maplimits',[zlim_min,zlim_max],'whitebk','on')
    if idx_lag == 1
        title('S - Sp', 'FontSize', FontSizeTitle) 
    end
%     c = colorbar;
%     c.Orientation = 'horizontal';
%     c.Location = 'westoutside';
%     c.Label.String = ['weights'];   
%     c.FontSize = FontSize;
%     c.Ticks = [round(zlim_min,2), round(zlim_max,2)];
end

set(gcf,'color','w');
