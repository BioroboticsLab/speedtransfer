function [sine_fit] = fit_sine(x, y, period_length)

fit_type_string = sprintf('y_offset+sin((2*pi/%d)*(x - x_offset))*y_scale', period_length);

ft = fittype(fit_type_string,'coefficients',{'x_offset','y_offset','y_scale'});

sine_fit = fit(x, y, ft, 'StartPoint',[0,0,0]);