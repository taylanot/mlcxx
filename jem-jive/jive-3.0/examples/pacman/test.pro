
log =
{
  //pattern = "*.info | *.debug | ! view.* | ! linsolve.*";
  pattern = "*";
};


control =
{
  pause    = 0.2;
  cmdFile  = "-";
  runWhile = true;
};


storageMode = "Compact";


meshgen =
{
  elemSize = 0.05;

  geometry =
  {
    type        = "Ring";
    innerRadius = 0.2;
    outerRadius = 1.0;
  };
};


input =
{
  file = "$(CASE_NAME).data";
};


init =
{
  vectors = [ "state = 1.0" ];
};


shapeTable =
{
  type         = "Auto";
  maxPrecision = 4;
};


output =
{
  file     = "$(CASE_NAME).out.gz";
  saveWhen = "save
                last   = -1;
              let
                retval = (i > last);
                last   = i
              return retval";
  vectors  = [ "state = state[%i]" ];
};


model =
{
  type   = "Matrix";

  matrix =
  {
    type           = "FEM";
    findSuperNodes = false;
    minBlockSize   = 1;
    symmetric      = true;
  };

  model =
  {
    type   = "Multi";
    models = [ "transport", "load" , "source" ];

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

        "let t = time return 3.0 *
           10 * exp( -3.0 * sqrt( (x - 0.5 * cos(t))^2 +
                                  (y - 0.5 * sin(t))^2 ) )";
    };
  };
};


linsolve =
{
  solver =
  {
    type          = "Verbose";
    printInterval = 1.0;

    solver =
    {
      type        = "GCR";
      precon.type = "ILUn";
      precision   = 1.0e-3;
    };
  };
};


park3 =
{
  deltaTime = 0.2;
};


pacman =
{
  elements = "lunch";
};


extra =
{
  modules = [ "graph", "sample" ];

  graph =
  {
    type     = "Graph";
    dataSets = [ "iter" ];

    iter =
    {
      key        = "Solver iterations";
      lineWidth  = 2.2;
      lineColor  = "blue";
      xData      = "i";
      yData      = "linsolve.iterCount";
      pointStyle = "circle";
    };
  };

  sample =
  {
    type = "Sample";
    file = "sample.out";
    dataSets = [ "i", "linsolve.iterCount" ];
  };
};


view =
{
  snapFile = "$(CASE_NAME)%2i.png";

  updateWhen  = true;

  window =
  {
    bgColor = "steel";
    width   = 400;
    height  = 400;
  };

  lights =
  {
    light0.enable    = true;
    light0.direction = [ 0.0, 0.0, -1.0 ];
    light0.color     = "white49";
    light1.enable    = false;
    light1.color     = "white49";
    light2.enable    = true;
    light2.color     = "white";
    light2.direction = [ -1.0, 0.0, 0.0 ];
    ambient          = "black";
  };

  palettes =
  {
    linear.minColor = [ 0.0, 0.0, 1.0, 0.2 ];
    linear.maxColor = [ 1.0, 0.0, 0.0, 0.9 ];
  };

  dataSets  = [ "state", "flux" ];

  state =
  {
    type   = "Vector";
    vector = "nodes/state";
  };

  flux  =
  {
    type  = "Table";
    table = "nodes/flux";
  };

  mesh    =
  {
    color      = "gold";
    lineWidth  = 1.0;
    shadeModel = "Smooth";
    showEdges  = false;

    deformation =
    {
      dz        = "state";
      autoScale = false;
      scale     = 0.2;
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
