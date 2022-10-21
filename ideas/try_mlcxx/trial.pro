control =
{
  runWhile = "i<1";
};

data =
{
  type = "linear";
  N = 100;
  ampl = 1.;
};

log =
{
  pattern = "*.info";
  file    = "-$(CASE_NAME).log";
};
