
log =
{
  file = "-$(CASE_NAME).log";
};

control =
{
  runWhile = "i < 1";
  fgMode   = true;
};

meshgen =
{
  elemSize = 0.5;

  geometry = "Ring"
  {
    innerRadius = 1.0;
    outerRadius = 5.0;
    elemType    = "Triangle6";
  };
};

input =
{
  file = "$(CASE_NAME).dat";
};

model = "Matrix"
{
  matrix = "FEM"
  {
    symmetric = true;
  };

  model = "Transport"
  {
    elements  = "all";
    diffusion = 1.0;
  };
};

linsolve =
{
  solver = "SparseLU"
  {};
};

shapeTable = "Auto"
{};

view =
{
  window =
  {
    title = "FemView";
  };

  camera =
  {
    // Increase the quality of the image by applying multi-sampling
    // (also known as anti aliasing). The higher the number, the
    // better the quality.

    quality = 5;
  };

  // Define two data sets that are used below to modify the way that
  // the mesh is rendered.

  dataSets = [ "state", "flux" ];

  // A data set that refers to the state vector.

  state = "Vector"
  {
    vector = "state";
  };

  // This data set that refers to a table containing the flux
  // in the nodes. This table is filled by the model.

  flux = "Table"
  {
    table = "nodes/flux";
  };

  mesh =
  {
    // Use the "state" data set to apply a deformation to the mesh.
    // In this way the solution is displayed as a displacement in
    // the Z-direction.

    deformation =
    {
      dz    = "state";
      scale = 0.5;
    };

    plugins = [ "colors", "nodes" ];

    // Use the "flux" data set to color the mesh.

    colors = "MeshColorView"
    {
      data  = "flux{ abs( $$x ) }";
    };

    // Display the nodes at the boundaries of the mesh.

    nodes = "MeshNodeView"
    {
      nodes = [ "innerEdge:numbers", "outerEdge:orbs" ];
    };
  };
};
