
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
  type      = "Example";
  parameter = 2.0;
};
