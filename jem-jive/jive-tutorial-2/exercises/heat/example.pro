
// This controls the standard output.

log =
{
  pattern = "*.info|*.debug";
};

// This controls the duration of the simlation.

control =
{
  pause   = 0.1;
  cmdFile = "-";
  runWhile = "i < 20";
};

// This controls the generation of the finite element mesh.

meshgen =
{
  elemSize = 0.05;

  geometry =
  {
    type        = "Ring";
    innerRadius = 0.1;
    outerRadius = 1.0;
    innerNodes  = 32;
  };
};

// This controls the input from other data files.

input =
{
  file = "$(CASE_NAME).data";
};

// This controls the initial condition.

init =
{
  vectors = [ "oldState = iniState" ];
};

// This associates a Shape with each element in the mesh.

shapeTable =
{
  type         = "Auto";
  maxPrecision = 4;
};

// This controls the output.

output =
{
  file     = "$(CASE_NAME).out.gz";
  saveWhen = "(i % 20) < 0.1";
  vectors  = [ "nodes/state" ];
};

// This is the model definition.

model =
{
  type   = "Matrix";
  linear = true;

  matrix =
  {
    type      = "FEM";
    symmetric = true;
  };

  model =
  {
    type   = "Multi";
    models = [ "transport", "load", "source" ];

    // The definition of the heat transport model.

    transport =
    {
      type      = "Transport";
      elements  = "all";
      precision = 2;

      diffusion = [ 1.0, 1.0 ];
    };

    // The definition of the time-dependent boundary conditions.

    load   =
    {
      type      = "LoadScale";
      scaleFunc = "sin(time)";

      model     =
      {
        type     = "Constraints";
        conTable = "Prescribed";
      };
    };

    // The definition of the heat source.

    source =
    {
      type        = "Source";
      elements    = "all";
      sourceFunc  =

        "10 * exp( -2 * sqrt( (x - 0.5 * cos(time))^2 +
                              (y - 0.5 * sin(time))^2 ) )";
    };
  };
};

// This constrols the linear solver.

linsolve =
{
  solver =
  {
    type        = "AGMRES";
    noiseLevel  = 1;
    precision   = 1.0e-1;
    maxVectors  = 200;
    precon.type = "Diag";
  };
};

// This controls the time integration module.

park3 =
{
  deltaTime = 0.1;
};

// This controls the graphical output.

//view =
//{
//  snapWhen   = false;
//  snapFile   = "$(CASE_NAME)%2i.png";
//
//  window =
//  {
//    bgColor = "steel";
//    width   = 400;
//    height  = 400;
//  };
//
//  palettes =
//  {
//    linear.minColor = [ 0.0, 0.0, 1.0, 0.2 ];
//    linear.maxColor = [ 1.0, 0.0, 0.0, 0.9 ];
//  };
//
//  dataSets  = [ "state", "flux" ];
//
//  state =
//  {
//    type  = "Vector";
//    vector = "state";
//  };
//
//  flux  =
//  {
//    type  = "Table";
//    table = "nodes/flux";
//  };
//
//  mesh    =
//  {
//    color      = "gold";
//    lineWidth  = 0.8;
//    shadeModel = "Smooth";
//    showEdges  = true;
//
//    deformation =
//    {
//      dz        = "state{ $$u * ($$u >= 0.2) + 0.2 * ($$u < 0.2) }";
//      autoScale = false;
//    };
//
//    plugins = [ "colors" ];
//
//    colors      =
//    {
//      type      = "MeshColorView";
//      data      = "flux { sqrt( $$x^2 + $$y^2 ) }";
//      palette   = "rainbow";
//      scaleStep = 0.2;
//    };
//  };
//};

// Extra modules.

//extra =
//{
//  modules = [ "graph" ];
//
//  // This displays the number of solver iterations in a graph.
//
//  graph = "Graph"
//  {
//    dataSets = [ "solverIter" ];
//
//    solverIter =
//    {
//      xData = "i";
//      yData = "linsolve.iterCount";
//    };
//  };
//};
