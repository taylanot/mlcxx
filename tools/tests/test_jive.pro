log =
{
  pattern   = "*.info";
  file      = "-$(CASE_NAME).log";
};

control =
{
  runWhile  = "i<2";
  fgMode    = false;
};

model =
{
  type      = "TestModel";
  parameter = 2.0;
};

userinput =
{
  modules = ['test', 'input'];

  test =
  {
    type    = 'TestModule';
  };

  input =
  {
    type    = 'input';
  };
  
};


