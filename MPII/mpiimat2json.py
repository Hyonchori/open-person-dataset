function
mpii_convert_json()
% convert
mpii
annotations.mat
file
to.json

% % load
annotation
file
fprintf('Load annotations... ')
data = load('/media/HDD2/Datasets/Human_Pose/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat');
fprintf('Done.\n')
% % open
file
fprintf('Open file mpii_human_pose_annotations.json\n')
path = '/media/HDD2/Datasets/Human_Pose/mpii/mpii_human_pose_v1_u12_2/';
fileID = fopen(strcat(path, 'mpii_human_pose_annotations.json'), 'w');

% % cycle
all
data and save
to
file(.json)
fprintf(fileID, '"{');

% annolist
fprintf(fileID, '\\"annolist\\":[');
for i=1:1: size(data.RELEASE.annolist, 2)
fprintf(fileID, '{');
% image
fprintf(fileID, '\\"image\\": {');
fprintf(fileID, '\\"name\\": \\"%s\\"', data.RELEASE.annolist(1, i).image.name);
fprintf(fileID, '},');

% annorect
fprintf(fileID, '\\"annorect\\": [');
for j=1:1: size(data.RELEASE.annolist(1, i).annorect, 2)
fprintf(fileID, '{');

flag_comma = 0;
% x1
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'x1')
    fprintf(fileID, '\\"x1\\": %d', data.RELEASE.annolist(1, i).annorect(j).x1);
    flag_comma = 1;
end

% y1
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'y1')
    if flag_comma, fprintf(fileID, ','); end
    flag_comma = 1;
    fprintf(fileID, '\\"y1\\": %d', data.RELEASE.annolist(1, i).annorect(j).y1);
end

% x2
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'x2')
    if flag_comma, fprintf(fileID, ','); end
    flag_comma = 1;
    fprintf(fileID, '\\"x2\\": %d', data.RELEASE.annolist(1, i).annorect(j).x2);
end

% y2
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'y2')
    if flag_comma, fprintf(fileID, ','); end
    flag_comma = 1;
    fprintf(fileID, '\\"y2\\": %d', data.RELEASE.annolist(1, i).annorect(j).y2);
end

% annopoints
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'annopoints')
    if flag_comma, fprintf(fileID, ','); end
    flag_comma = 1;
    if isempty(data.RELEASE.annolist(1, i).annorect(j).annopoints)
        fprintf(fileID, '\\"annopoints\\": []');
    else
        % disp(i)
        fprintf(fileID, '\\"annopoints\\": {\\"point\\": [');
        for d=1:1: size(data.RELEASE.annolist(1, i).annorect(j).annopoints.point, 2)
        fprintf(fileID, '{');
        fprintf(fileID, '\\"x\\": %d, ', data.RELEASE.annolist(1, i).annorect(j).annopoints.point(1, d).x);
        fprintf(fileID, '\\"y\\": %d, ', data.RELEASE.annolist(1, i).annorect(j).annopoints.point(1, d).y);
        fprintf(fileID, '\\"id\\": %d ', data.RELEASE.annolist(1, i).annorect(j).annopoints.point(1, d).id);
        if isfield(data.RELEASE.annolist(1, i).annorect(j).annopoints.point(1, d), 'is_visible')
            fprintf(fileID, ',');
            if isempty(data.RELEASE.annolist(1, i).annorect(j).annopoints.point(1, d).is_visible)
                fprintf(fileID, '\\"is_visible\\": []');
            else
                fprintf(fileID, '\\"is_visible\\": %d',
                        data.RELEASE.annolist(1, i).annorect(j).annopoints.point(1, d).is_visible);
            end
        end
        fprintf(fileID, '}');

        if d < size(data.RELEASE.annolist(1, i).annorect(j).annopoints.point, 2)
            fprintf(fileID, ',');
        end
    end
    fprintf(fileID, ']}');
end
end

% scale
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'scale')
    if flag_comma, fprintf(fileID, ','); end
    flag_comma = 1;
    if isempty(data.RELEASE.annolist(1, i).annorect(j).scale)
        fprintf(fileID, '\\"scale\\": []');
    else
        fprintf(fileID, '\\"scale\\": %d', data.RELEASE.annolist(1, i).annorect(j).scale);
    end
end

% objpos
if isfield(data.RELEASE.annolist(1, i).annorect(j), 'objpos')
    if flag_comma, fprintf(fileID, ','); end
    flag_comma = 1;
    if isempty(data.RELEASE.annolist(1, i).annorect(j).objpos)
        fprintf(fileID, '\\"objpos\\": []');
    else
        fprintf(fileID, '\\"objpos\\": {');
        fprintf(fileID, '\\"x\\": %d, ', data.RELEASE.annolist(1, i).annorect(j).objpos.x);
        fprintf(fileID, '"y\\": %d', data.RELEASE.annolist(1, i).annorect(j).objpos.y);
        fprintf(fileID, '}');
    end
end

fprintf(fileID, '}');

if j < size(data.RELEASE.annolist(1, i).annorect, 2)
    fprintf(fileID, ',');
end
end
fprintf(fileID, '],');

% frame_sec
if isempty(data.RELEASE.annolist(1, i).frame_sec)
    fprintf(fileID, '\\"frame_sec\\": [], ');
else
    fprintf(fileID, '\\"frame_sec\\": %d, ', data.RELEASE.annolist(1, i).frame_sec);
end

% vididx
if isempty(data.RELEASE.annolist(1, i).vididx)
    fprintf(fileID, '\\"vididx\\": [] ');
else
    fprintf(fileID, '\\"vididx\\": %d ', data.RELEASE.annolist(1, i).vididx);
end

fprintf(fileID, '}');

if i < size(data.RELEASE.annolist, 2)
    fprintf(fileID, ',');
end
end
fprintf(fileID, '],');

% img_train
fprintf(fileID, '\\"img_train\\":[');
for i=1:1: size(data.RELEASE.img_train, 2)
fprintf(fileID, '%d', data.RELEASE.img_train(1, i));

if i < size(data.RELEASE.img_train, 2)
    fprintf(fileID, ',');
end
end
fprintf(fileID, '],');

% version
fprintf(fileID, '\\"version\\": %s,', data.RELEASE.version);

% single_person
fprintf(fileID, '\\"single_person\\":[');
for i=1:1: size(data.RELEASE.single_person, 1)
if isempty(data.RELEASE.single_person{i, 1})
fprintf(fileID, '[]');
elseif
size(data.RELEASE.single_person
{i, 1}, 1) == 1
fprintf(fileID, '%d', data.RELEASE.single_person
{i, 1});
else
fprintf(fileID, '[');
for j=1:1: size(data.RELEASE.single_person
{i, 1}, 1)
fprintf(fileID, '%d', data.RELEASE.single_person
{i, 1}(j));
if j < size(data.RELEASE.single_person{i, 1}, 1)
fprintf(fileID, ',');
end
end
fprintf(fileID, ']');
end

if i < size(data.RELEASE.single_person, 1)
    fprintf(fileID, ',');
end
end
fprintf(fileID, '],');

% act
fprintf(fileID, '\\"act\\":[');
for i=1:1: size(data.RELEASE.act, 1)
fprintf(fileID, '{');
% cat_name
if isempty(data.RELEASE.act(i, 1).cat_name)
    fprintf(fileID, '\\"cat_name\\": [], ');
else
    fprintf(fileID, '\\"cat_name\\": \\"%s\\", ', data.RELEASE.act(i, 1).cat_name);
end

% act_name
if isempty(data.RELEASE.act(i, 1).act_name)
    fprintf(fileID, '\\"act_name\\": [], ');
else
    fprintf(fileID, '\\"act_name\\": \\"%s\\", ', data.RELEASE.act(i, 1).act_name);
end

% act_id
fprintf(fileID, '\\"act_id\\": %d', data.RELEASE.act(i, 1).act_id);

fprintf(fileID, '}');

if i < size(data.RELEASE.act, 1)
    fprintf(fileID, ',');
end
end
fprintf(fileID, '],');

% video_list
fprintf(fileID, '\\"video_list\\":[');
for i=1:1: size(data.RELEASE.video_list, 2)
fprintf(fileID, '\\"%s\\"', data.RELEASE.video_list
{1, i});

if i < size(data.RELEASE.video_list, 2)
    fprintf(fileID, ',');
end
end
fprintf(fileID, ']');

% % close
file
fprintf(fileID, '}');
fclose(fileID);

fprintf('Script complete.\n')
end