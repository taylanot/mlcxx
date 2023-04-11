seed = 24;

data =
{
    type = "regression";

    genfunc = "Linear";

    Ntrn = 1000;
    Ntst = 10000;
    scale = 1.;
    phase = 0.;
    eps = 0.;
};




model = 
{
  type = "LeastSquaresRegression";

    
  hpt =
  {
    tune = true;

    lambda = 1.;
    bias = true;

    lambda_bounds = [0.,1.];
    bias_bounds = [true, false];

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
  type = "learningcurve";
  rep = 100;
  N_bounds = [5, 50];
  res = 10;
  //type = "train/test";

};


