
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jive/util/XTable.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Assignable.h>
#include <jive/geom/Quad.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/ElementGroup.h>
#include "import.h"
#include "declare.h"
#include "Array.h"

using jive::geom::IShape;
using jive::model::Model;
using jive::util::XTable;
using jive::util::XDofSpace;
using jive::util::Assignable;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::ElementGroup;
using jive::algebra::MatrixBuilder;

//=======================================================================
//   class ElasticModel
//=======================================================================

class ElasticModel : public Model
{
 public:

                          ElasticModel

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props,
      const Properties&     globdat );

  virtual bool            takeAction

    ( const String&         action,
      const Properties&     params,
      const Properties&     globdat );


 private:

  void                    assemble_

    ( MatrixBuilder&        mbld,
      const Vector&         fint,
      const Vector&         disp );

  void                    getStress_

    ( XTable&               table,
      const Vector&         weights,
      const Vector&         disp );

  void                    getStiffmat_

    ( const Matrix&         stiff )   const;

  static void             getBMatrix_

    ( const Matrix&         bmat,
      const Matrix&         grads );


 private:

  Assignable
    <ElementGroup>        group_;
  Assignable<NodeSet>     nodes_;
  Assignable<ElementSet>  elems_;
  Ref<XDofSpace>          dofs_;
  Ref<IShape>             shape_;
  IdxVector               jtypes_;

  double                  young_;
  double                  poisson_;

};

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

ElasticModel::ElasticModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Model ( name )

{
  using jive::geom::Quad4;

  Properties  myProps = props.findProps ( myName_ );
  Properties  myConf  = conf .makeProps ( myName_ );
  String      ischeme = "Gauss2 * Gauss2";
  String      bscheme = "Gauss2";


  // Get the elements and nodes from the global database.

  group_ = ElementGroup::get  ( myConf,  myProps,
                                globdat, getContext() );
  elems_ = group_.getElements ();
  nodes_ = elems_.getNodes    ();

  // Initialize the internal shape.

  shape_ = Quad4::getShape ( "shape", ischeme, bscheme );

  // Check whether the mesh is valid.

  JEM_PRECHECK ( nodes_.rank() == 2 );

  elems_.checkElements ( getContext(), shape_->nodeCount() );

  // Define the DOFs.

  jtypes_.resize ( 2 );

  dofs_      = XDofSpace::get ( nodes_.getData(), globdat );
  jtypes_[0] = dofs_->addType ( "dx" );
  jtypes_[1] = dofs_->addType ( "dy" );

  dofs_->addDofs (
    elems_.getUniqueNodesOf ( group_.getIndices() ),
    jtypes_
  );

  // Get the material parameters.

  myProps.get ( young_,   "young" );
  myProps.get ( poisson_, "poisson" );

  // Report the used material parameters.

  myConf.set ( "young",   young_ );
  myConf.set ( "poisson", poisson_ );
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

// Performs an action requested by a module or a parent model.

bool ElasticModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;
  using jive::model::StateVector;

  if ( action == Actions::GET_MATRIX0 )
  {
    Ref<MatrixBuilder>  mbld;
    Vector              fint;
    Vector              disp;

    // Get the action-specific parameters.

    params.get ( mbld, ActionParams::MATRIX0 );
    params.get ( fint, ActionParams::INT_VECTOR );

    // Get the current displacements.

    StateVector::get ( disp, dofs_, globdat );

    // Assemble the global stiffness matrix together with
    // the internal vector.

    assemble_ ( *mbld, fint, disp );

    return true;
  }

  if ( action == Actions::GET_TABLE )
  {
    Ref<XTable>  table;
    String       name;

    // Get the action-specific parameters.

    params.get ( table, ActionParams::TABLE );
    params.get ( name,  ActionParams::TABLE_NAME );

    // Check whether the requested table is supported by
    // this model.

    if ( (name == "stress") &&
        (table->getRowItems() == nodes_.getData()) )
    {
      Vector  weights, disp;

      params      .get ( weights, ActionParams::TABLE_WEIGHTS );
      StateVector::get ( disp,    dofs_, globdat );

      getStress_ ( *table, weights, disp );

      return true;
    }
  }

  if ( action == Actions::COMMIT )
  {
    Vector  disp;

    StateVector::get ( disp, dofs_, globdat );

    // Store the maximum displacement in the global database.

    globdat.set ( "var.disp",  max( disp ) );

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   assemble_
//-----------------------------------------------------------------------

// Assembles the global stiffness matrix and the internal force vector.

void ElasticModel::assemble_

  ( MatrixBuilder&  mbld,
    const Vector&   fint,
    const Vector&   disp )

{
  using jem::numeric::matmul;
  using jem::numeric::MatmulChain;

  const idx_t   rank      = nodes_.rank         ();
  const idx_t   ipCount   = shape_->ipointCount ();
  const idx_t   nodeCount = shape_->nodeCount   ();
  const idx_t   dofCount  = rank * nodeCount;

  IdxVector     ielems    = group_.getIndices   ();

  Cubix         grads     ( rank, nodeCount, ipCount );
  Matrix        bmat      ( 3,    dofCount );
  Matrix        stiff     ( 3,    3 );
  Vector        weights   ( ipCount );

  Matrix        elmat     ( dofCount, dofCount );
  Vector        eldisp    ( dofCount );

  Matrix        coords    ( rank, nodeCount );
  IdxVector     inodes    ( nodeCount );
  IdxVector     idofs     ( dofCount );

  MatmulChain
    <double,3>  mc3;


  // Get the strain-stress stiffness matrix.

  getStiffmat_ ( stiff );

  for ( idx_t ie = 0; ie < ielems.size(); ie++ )
  {
    // Get the global element index.

    idx_t  ielem = ielems[ie];

    // Get the element nodes, coordinates and DOF indices.

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, jtypes_ );

    // Get the element shape function gradients and integration
    // point weights.

    shape_->getShapeGradients ( grads, weights, coords );

    // Assemble the element matrix.

    elmat = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      getBMatrix_ ( bmat, grads[ip] );

      elmat += weights[ip] *

        mc3.matmul ( bmat.transpose(), stiff, bmat );
    }

    // Add the element matrix to the global stiffness matrix.

    mbld.addBlock ( idofs, idofs, elmat );

    // Compute the internal force vector. This is straightforward
    // because the model is linear.

    eldisp       = disp[idofs];
    fint[idofs] += matmul ( elmat, eldisp );
  }
}

//-----------------------------------------------------------------------
//   getStress_
//-----------------------------------------------------------------------

// Computes the average stresses in the nodes.

void ElasticModel::getStress_

  ( XTable&        table,
    const Vector&  weights,
    const Vector&  disp )

{
  using jem::ALL;
  using jem::numeric::matmul;

  const idx_t   rank      = nodes_.rank       ();
  const idx_t   nodeCount = shape_->nodeCount ();
  const idx_t   dofCount  = rank * nodeCount;

  IdxVector     ielems    = group_.getIndices ();

  Cubix         grads     ( rank, nodeCount, nodeCount );
  Matrix        bmat      ( 3,    dofCount );
  Matrix        stiff     ( 3,    3 );
  Matrix        stress    ( 3,    nodeCount );
  Vector        strain    ( 3 );
  Matrix        coords    ( rank, nodeCount );
  IdxVector     inodes    ( nodeCount );
  IdxVector     idofs     ( dofCount );
  IdxVector     jcols     ( 3 );
  Vector        eldisp    ( dofCount );


  jcols[0] = table.addColumn ( "sxx" );
  jcols[1] = table.addColumn ( "syy" );
  jcols[2] = table.addColumn ( "sxy" );

  getStiffmat_ ( stiff );

  for ( idx_t ie = 0; ie < ielems.size(); ie++ )
  {
    idx_t  ielem = ielems[ie];

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, jtypes_ );

    // Get the shape function gradients in the nodes of the
    // element and compute the stresses in those nodes.

    shape_->getVertexGradients ( grads, coords );

    eldisp = disp[idofs];

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      getBMatrix_ ( bmat, grads[i] );

      matmul ( strain,    bmat,  eldisp );
      matmul ( stress[i], stiff, strain );
    }

    // Update the table and the table weights.

    table.addBlock ( inodes, jcols, stress.transpose() );

    weights[inodes] += 1.0;
  }
}

//-----------------------------------------------------------------------
//   getStiffmat_
//-----------------------------------------------------------------------

void ElasticModel::getStiffmat_ ( const Matrix& stiff ) const
{
  JEM_PRECHECK ( stiff.size(0) == 3 &&
                 stiff.size(1) == 3 );

  const double  e  = young_;
  const double  nu = poisson_;
  const double  a  = e / (1.0 - nu * nu);

  stiff = 0.0;

  stiff(0,0) = a;
  stiff(0,1) = a * nu;
  stiff(1,0) = a * nu;
  stiff(1,1) = a;
  stiff(2,2) = a * 0.5 * (1.0 - nu);
}

//-----------------------------------------------------------------------
//   getBMatrix_
//-----------------------------------------------------------------------


void ElasticModel::getBMatrix_

  ( const Matrix&  bmat,
    const Matrix&  grads )

{
  JEM_PRECHECK ( bmat .size(0) == 3 &&
                 grads.size(0) == 2 &&
                 bmat .size(1) == 2 * grads.size(1) );

  const idx_t  nodeCount = grads.size (1);

  bmat = 0.0;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  i = 2 * inode;

    bmat(0,i + 0) = grads(0,inode);
    bmat(1,i + 1) = grads(1,inode);

    bmat(2,i + 0) = grads(1,inode);
    bmat(2,i + 1) = grads(0,inode);
  }
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newElasticModel
//-----------------------------------------------------------------------

// Creates a new ElasticModel instance.

Ref<Model>           newElasticModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::newInstance;

  return newInstance<ElasticModel> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareElasticModel
//-----------------------------------------------------------------------

// Registers the ElasticModel class with the ModelFactory.

void declareElasticModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "Elastic", newElasticModel );
}

