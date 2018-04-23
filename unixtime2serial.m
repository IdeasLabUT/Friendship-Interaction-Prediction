function date_sn = unixtime2serial(unix_time)
% unixtime2serial(unix_time) converts a timestamp in Unix time format
% (based on seconds since standard epoch of 1970/1/1) into a MATLAB serial
% date number.
% Author: Kevin Xu

offset = 719529;	% 1970/1/1 is 719529 in serial date number
secs_in_day = 86400;	% Number of seconds in a day
date_sn = unix_time/secs_in_day + offset;