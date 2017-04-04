function [ output_args ] = import( prefix, b )


pad1x = dlmread(strcat(prefix, '_padding1_x.txt'));
pad1y = dlmread(strcat(prefix, '_padding1_y.txt'));
pad1dx = dlmread(strcat(prefix, '_padding1_dx.txt'));
pad1dy = dlmread(strcat(prefix, '_padding1_dy.txt'));

end

