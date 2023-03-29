
control =
{
  runWhile = "i < 100";
  pause    = 0.1;
};

model =
{
  type = "Example";
};

graph =
{
  dataSets = [ "data" ];
  data     =
  {
    // Plot the runtime variable "v" as function of the
    // iteration number.

    xData  = "i";
    yData  = "v";
  };
};

sample =
{
  // Store two columns in the output file: one containing
  // the iteration number and one containing the runtime
  // variable "v".

  file      = "runvar.out";
  separator = ",";
  dataSets  = [ "i", "v" ];
};
