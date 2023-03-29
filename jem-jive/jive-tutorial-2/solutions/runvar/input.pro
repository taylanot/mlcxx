
control =
{
  runWhile = "v > 0";
};

model =
{
  type = "Example";
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
