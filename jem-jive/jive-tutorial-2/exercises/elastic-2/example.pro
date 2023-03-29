
// Parameters controlling the output messages.

log =
{
  // Print all info and debug messages.

  pattern = "*.info | *.debug";

  // Print to the terminal and to a file.

  file    = "-$(CASE_NAME).log";
};


// Parameters for the InputModule.

input =
{
  file = "$(CASE_NAME).dat";
};


// Parameters for the ControlModule.

control =
{
  runWhile = "i < 10";
};


// Parameters controlling the LinsolveModule.

linsolve =
{
  solver = "SparseLU"
  {};
};


// Parameters controlling the output.

output =
{
  file    = "$(CASE_NAME).out";

  // Save the state vector and label it "disp" in the
  // output file.

  vectors = [ "state = disp" ];

  // Save the solution every iteration.

  saveWhen = true;
};


// The definition of the model tree.

model = "Matrix"
{
  matrix =
  {
    type = "FEM";  // Use the FEMatrixBuilder.
  };

  model  = "Multi"
  {
    models = [ "elastic", "load" ];

    elastic = "Elastic"
    {
      young = 0.0001;
      area  = 200e9;
    };

    // Define a point load at the right end of the bar.
    // The load increases each iteration.

    load = "LoadScale"
    {
      scaleFunc = "1 + i";
      model     = "PointLoad"
      {
        // Use the table defined in the data file.

        loadTable = "load";
      };
    };
  };
};
