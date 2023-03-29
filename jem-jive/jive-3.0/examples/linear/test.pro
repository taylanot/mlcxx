
storageMode = "Compact";

log =
{
  pattern = "*.info|*.debug";
  file    = "-$(CASE_NAME).log";
  rank    = 0;
};


checkpoint =
{
  file     = "";
  backups  = 2;
};


control =
{
  runWhile = "i < 2";
};


meshgen =
{
  elemSize = 0.001;

  geometry =
  {
    type = "Rectangle";
    size = [ 2.0, 1.0 ];
  };
};


input =
{
  file = "$(CASE_NAME).data";
};


shapeTable =
{
  type         = "Auto";
  maxPrecision = 4;
};


model =
{
  type  = "MP";
  model =
  {
    type   = "Matrix";

    matrix =
    {
      type      = "FEM";
      symmetric = true;
    };

    model =
    {
      type   = "Multi";
      models = [ "transport" ];

      transport =
      {
        type = "Debug";

        model =
        {
          type      = "Transport";
          elements  = "all";
          precision = 1;
          diffusion = 1.0;
        };
      };
    };
  };
};


mpart =
{
  overlap     = 1;
  mapFile     = "$(CASE_NAME).map.gz";
  checkBlocks = true;
};


linsolve =
{
  solver = "GMRES"
  {
    precon = "Dual"
    {
      precon1 = "ILUd"
      {
        reorder = true;
      };
      precon0 = "Coarse"
      {
        restrictor = "RigidBody"
        {
        };
      };
    };
    noiseLevel    = 2;
    printInterval = 0.1;
  };
};


output =
{
  file     = "$(CASE_NAME).out";
  append   = false;

  vectors  = [ "state = temp[%i]" ];
  tables   = [ "nodes/flux  = flux[%i]" ];
  saveWhen = true;
};
