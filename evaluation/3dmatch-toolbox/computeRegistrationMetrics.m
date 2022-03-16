% Arguments: external_path gt_log_path gt_info_path result_log_path

arg_list = argv ();

% Add external_path
addpath(arg_list{1});

% Compute registration error
gt = mrLoadLog(fullfile(arg_list{2}));
gt_info = mrLoadInfo(fullfile(arg_list{3}));
result = mrLoadLog(fullfile(arg_list{4}));
[recall,precision] = mrEvaluateRegistration(result,gt,gt_info);
printf("%f\n%f\n", recall, precision);
