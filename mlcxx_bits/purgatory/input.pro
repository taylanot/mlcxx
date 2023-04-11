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



objective =
{
  type = "train/test";
};

model = 
{
  type = "LinearRegression";

  tune =
  {
    cv = 
    {
      type = "SimpleCV";
      valid = 0.2;
    };

    loss = "MSE";
    
    hyper =
    {
      lambda = [0.,1.];
      bias = [true, false];
    };  

      res = 100;
  };

  hyper=
  {
    lambda = 1.;
    bias = true;
  };

};

