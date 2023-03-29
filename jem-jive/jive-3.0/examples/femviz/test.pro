

log =
{
  pattern = "*.info | *.debug";
};


input =
{
  file = "test.data";
};


shapeTable =
{
  type           = "Auto";
  interfaceElems = "interfaces";
};


view =
{
  mesh.color = [ 0.0, 0.0, 1.0, 0.2 ];

  dataSets = "disp";

  camera =
  {
    quality = 8;
  };

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

  plugins = [ "nodes", "input", "mesh2" ];

  nodes =
  {
    type    = "PointView";
    points  = "nodes";
    markers = [ "upper:tee45" ];

    displacements = "disp";
  };

  input =
  {
    type = "MeshReader";
    file = "test.data";
    mesh = "test";

    shapeTable =
    {
      type = "Auto";
      interfaceElems = "interfaces";
    };
  };

  mesh2 =
  {
    type = "MeshView";
    elements = "test.elements/all";
  };
};
