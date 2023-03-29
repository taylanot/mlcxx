
// Parameters for the InputModule.

input =
{
  file = "input.dat";
};

// Parameters for the ControlModule.

control =
{
  runWhile = "save
                step = 0
              let
                step = step + 1;
              return
                step < 3";
};

model =
{
  type   = "Multi";
  models = [ "first", "second", "third" ];

  first  =
  {
    type      = "Example";
    parameter = 1.0;
  };

  second =
  {
    type      = "Example";
    parameter = 2.0;
  };

  third =
  {
    type      = "LoadScale";
    scaleFunc = "10.0 * i";
    model     =
    {
      type      = "Example";
      parameter = 3.0;
    };
  };
};
