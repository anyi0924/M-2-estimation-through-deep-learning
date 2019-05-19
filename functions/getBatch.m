function y = getBatch(imdb, batch, opts )

im = single(imdb.images(:,:,:,batch));
gtdata = imdb.coslabel(:,:,:,batch);


im = im-opts.imgRbgMean.r;

if opts.useGpu
  im = gpuArray(im);
end

y = {'input', im, 'label', gtdata} ;
