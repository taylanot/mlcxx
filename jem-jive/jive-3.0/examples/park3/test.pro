
log =
{
  pattern = "*.info|*.debug";
};


control =
{
  pause   = 0.1;
  cmdFile = "-";
  runWhile = "i < 200";
};


storageMode = "Compact";


meshgen =
{
  elemSize = 0.006;

  geometry =
  {
    type        = "Ring";
    innerRadius = 0.1;
    outerRadius = 1.0;
    innerNodes  = 32;
  };
};


input =
{
  file = "$(CASE_NAME).data";
};


init =
{
  vectors = [ "oldStateii = iniState" ];
};


shapeTable =
{
  type         = "Auto";
  maxPrecision = 4;
};


output =
{
  file     = "$(CASE_NAME).out.gz";
  interval = 20;
  vectors  = [ "nodes/state" ];
};


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

    transport =
    {
      type      = "Transport";
      elements  = "all";
      precision = 2;

      diffusion = [ 1.0, 1.0 ];
    };

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

/*
linsolve =
{
  solver =
  {
    type = "Schur";
    noiseLevel = 1;

    outerSolver =
    {
      type = "AGMRES";
      noiseLevel = 1;
      precon.type = "Neumann";
    };
  };
};
*/


linsolve =
{
  solver =
  {
    type = "AGMRES";
    noiseLevel = 1;
    precision = 1.0e-1;
    maxVectors = 200;
    precon.type = "Diag";
  };
};


park3 =
{
  deltaTime = 0.1;
};


view =
{
  snapWhen   = false;
  snapFile   = "$(CASE_NAME)%2i.png";

  window =
  {
    bgColor = "steel";
    width   = 400;
    height  = 400;
  };

  camera.quality = 1;

  lights =
  {
    light2.enable = true;
    light3.enable = false;
  };

  palettes =
  {
    linear.minColor = [ 0.0, 0.0, 1.0, 0.2 ];
    linear.maxColor = [ 1.0, 0.0, 0.0, 0.9 ];
  };

  dataSets  = [ "state", "flux" ];

  state =
  {
    type  = "Vector";
    items = "nodes";
  };

  flux  =
  {
    type  = "Table";
    items = "nodes";
  };

  mesh    =
  {
    color      = "gold";
    lineWidth  = 0.8;
    shadeModel = "Smooth";
    showEdges  = true;

    deformation =
    {
      dz        = "state{ $$u * ($$u >= 0.2) + 0.2 * ($$u < 0.2) }";
      autoScale = false;
    };

    plugins = [ "colors" ];

    colors      =
    {
      type      = "MeshColorView";
      data      = "flux { sqrt( $$x^2 + $$y^2 ) }";
      palette   = "rainbow";
      scaleStep = 0.2;
    };
  };
};


