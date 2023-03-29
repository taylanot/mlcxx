
// Parameters controlling the output messages.

log =
{
  // Print to the terminal and to a file.

  file = "-$(CASE_NAME).log";
};


// Parameters for the MeshgenModule.

meshgen =
{
  geometry = none;
};

// Parameters for the InputModule.

input =
{
  file = "$(CASE_NAME).dat";
};


// Parameters for the ControlModule.

control =
{
  pause    = 0.2;
  runWhile = "i < 10";
  fgMode   = true;
};


// Parameters controlling the NonlinModule.

nonlin =
{
  maxIter   = 1000;
  precision = 1.0e-4;
  solver    = "SparseLU"
  {};
};


// Parameters controlling the output.

output =
{
  file    = "$(CASE_NAME).out";

  // Save the state vector and label it "disp" in the
  // output file.

  vectors = [ "state = disp" ];

  // Save the average stresss and the damage in the elements.

  tables  = [ "elements/stress",
              "elements/damage" ];

  // Save the solution every 5 load steps.

  saveWhen = "(i % 5) < 0.1";
};


// Parameters controlling the GraphModule.

graph =
{
  graph.title   = "Load - Displacement";
  graph.xLabel  = "Displacement";
  graph.yLabel  = "Load";
  graph.showKey = false;
  window.title  = graph.title;

  dataSets = [ "load" ];

  load =
  {
    xData     = "disp";
    yData     = "load";
    lineWidth = 3.0;
  };
};


// Parameters for the FemView module.

view =
{
  window.title = "Damage";

  dataSets = [ "disp", "damage" ];

  disp = "Vector"
  {
    vector = "state";
  };

  damage = "Table"
  {
    table = "elements/damage";
  };

  mesh =
  {
    deformation = "10 * disp";

    plugins = [ "colors", "nodes" ];

    colors = "MeshColorView"
    {
      data = "damage";
    };

    nodes = "MeshNodeView"
    {
      nodes = [ "leftCorner:discs",
                "rightCorner:discs",
                "loadPoint:diamonds" ];
    };
  };
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
    models = [ "damage", "load" ];

    damage = "Damage"
    {
      elements = "all";
      young    = 200e9;
      poisson  = 0.2;
      alpha    = 0.001;
      beta     = 0.002;
    };

    load = "LoadScale"
    {
      scaleFunc = "2 + i";
      model     = "Constraints"
      {
        // Use the table defined in the data file.

        conTable = "disp";
      };
    };
  };
};


