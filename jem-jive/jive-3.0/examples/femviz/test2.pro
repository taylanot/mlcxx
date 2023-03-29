
log =
{
  pattern = "*.info | *.debug";
};


input =
{
  file = "test2.data";
};


shapeTable =
{
  type           = "Auto";
  interfaceElems = "interfaces";
};


view =
{
  mesh =
  {
    deformation = "disp";

    plugins = [ "colors" ];

    colors      =
    {
      type      = "MeshColorView";
      data      = "temp[T]";
      palette   = "rainbow";
      scaleStep = 0.2;
    };
  };

  dataSets = [ "disp", "temp" ];

  markerStyles =
  {
    styles = [ "tee45" ];

    tee45 =
    {
      type  = "Tee";
      angle = 45.0;
    };
  };

  disp     =
  {
    type   = "Table";
    table  = "nodes/disp";
  };

  temp    =
  {
    type  = "Table";
    table = "nodes/temp1";
  };
};
