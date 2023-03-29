
log =
{
  pattern = "*.info|*.debug";
  file    = "-$(CASE_NAME).log";
  rank    = 0;
};


input =
{
  file = "$(CASE_NAME).data";
};


model =
{
  type   = "Matrix";

  matrix =
  {
    type = "FEM";
  };

  model =
  {
    type   = "Poisson";
  };
};


linsolve =
{
  solver =
  {
    type       = "SkylineLU";
    useThreads = true;
  };
};


output =
{
  file    = "$(CASE_NAME).out";
  append  = false;

  vectors = [ "state = solution" ];
  tables  = [ "elements/flux = flux[%i]" ];
};
