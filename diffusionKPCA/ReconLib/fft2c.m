function res = fft2c(x)
fctr = size(x,1)*size(x,2);
for n=1:size(x,3)
	res(:,:,n) = 1/sqrt(fctr)*fft2(x(:,:,n));
end


