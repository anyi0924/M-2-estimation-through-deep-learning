classdef myLoss < dagnn.Loss
 
  properties (Transient)
    clip_d = 1
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      d_errors = gather(inputs{1})-(inputs{2});
      fs = size(inputs{1},3);
      bs = size(inputs{1},4);
      outputs{1} = sum((sum(d_errors.^2)));
      n = obj.numAveraged ;
      m = n + fs*bs ;
      obj.average = (n * obj.average + outputs{1}) / m ;
      obj.numAveraged = m ;      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      d_errors = gather(inputs{1})-(inputs{2});
%       d_errors(d_errors > obj.clip_d) = obj.clip_d; 
%       d_errors(d_errors < -obj.clip_d) = -obj.clip_d;
      fs = size(inputs{1},3);
      dx = 2*d_errors/fs;
     % dx(:,:,6:9,:) = 5*dx(:,:,6:9,:);
   
      dx = gpuArray(dx);
      derInputs{1} = dx;
      derInputs{2} = [] ;
      derParams = {} ;
      
    end
    
    function setClip(obj, rd)
        obj.clip_d = rd;
    end
    
    function obj = myLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
