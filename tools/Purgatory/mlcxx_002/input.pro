seed = 24;

data =
{
    type = "regression";

    genfunc = "Sine";

    Ntrn = 1000;
    Ntst = 10000;
    scale = 1.;
    phase = 0.;
    eps = 0.;
};




model = 
{
  type = "KernelLeastSquaresRegression";
  
  kernel = 
  {
    type = "Gaussian";
    param_bounds = [0.001, 10.];
  };
    
  hpt =
  {
    tune = true;

    lambda = 1.;

    lambda_bounds = [0.,1.];

    res = 100;

    cv = 
    {
      type = "SimpleCV";
      param = 0.2;
    };

  };  

};

objective =
{
  //type = "learningcurve";
  rep = 100;
  N_bounds = [5, 50];
  res = 10;
  type = "train/test";

};

