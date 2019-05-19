classdef myAccuracy < dagnn.Loss
    
  properties (Transient)
    L1_error = 0
    L2_error = 0
  end
    
  methods
     function outputs = forward(obj, inputs, params)
        d_errors = gather(inputs{1}) - inputs{2};
        bs = size(inputs{2},4);
        
        L1_error_tmp = sum(sum(abs(d_errors)));
        L2_error_tmp = sum(sqrt(sum(d_errors.^2)));
              
        obj.L1_error = obj.L1_error + L1_error_tmp;
        obj.L2_error = obj.L2_error + L2_error_tmp;
        
        obj.numAveraged = obj.numAveraged + bs;
        outputs{1} = obj.average ;
        
     end
     
     function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs{1} = [] ;
        derInputs{2} = [] ;
        derParams = {} ;
     end
     
     function reset(obj)
        obj.L1_error = 0 ;
        obj.L2_error = 0 ;
        obj.average = [0;0] ;
        obj.numAveraged = 0 ;
     end
     
     function recalerror(obj)
        obj.L1_error = obj.L1_error/obj.numAveraged;
        obj.L2_error = obj.L2_error/obj.numAveraged;
        obj.average = [obj.L1_error; obj.L2_error] ;
     end
     
     function str = toString(obj)
        str = sprintf('L1_error:%.3f, L2_error:%.3f', ...
                    obj.L1_error, obj.L2_error) ;
     end
      
     function obj = myAccuracy(varargin)
        obj.load(varargin) ;
     end
   end
    
    
end

