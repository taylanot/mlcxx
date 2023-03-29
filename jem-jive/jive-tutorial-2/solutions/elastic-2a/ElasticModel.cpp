
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Assignable.h>
#include <jive/geom/Line.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include "import.h"
#include "declare.h"
#include "Array.h"

using jive::geom::IShape;
using jive::model::Model;
using jive::util::XDofSpace;
using jive::util::Assignable;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
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


 private:

  Assignable<NodeSet>     nodes_;
  Assignable<ElementSet>  elems_;
  Ref<XDofSpace>          dofs_;
  Ref<IShape>             shape_;
  idx_t                   jtype_;

  double                  young_;
  double                  area_;

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
  using jive::geom::Line2;

  Properties  myProps = props.findProps ( myName_ );
  Properties  myConf  = conf .makeProps ( myName_ );
  String      ischeme = "Gauss2";


  // Get the elements and nodes from the global database.

  elems_ = ElementSet::get ( globdat, getContext() );
  nodes_ = elems_.getNodes ();

  // Initialize the internal shape.

  shape_ = Line2::getShape ( "shape", ischeme );

  // Check whether the mesh is valid.

  JEM_PRECHECK ( nodes_.rank() == 1 );

  elems_.checkElements ( getContext(), shape_->nodeCount() );

  // Define the DOFs.

  dofs_  = XDofSpace::get ( nodes_.getData(), globdat );
  jtype_ = dofs_->addType ( "u" );

  for ( idx_t inode = 0; inode < nodes_.size(); inode++ )
  {
    dofs_->addDof ( inode, jtype_ );
  }

  // Get the material parameters.

  myProps.get ( young_, "young" );
  myProps.get ( area_,  "area" );

  // Report the used material parameters.

  myConf.set ( "young", young_ );
  myConf.set ( "area",  area_ );
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

  const idx_t  rank      = nodes_.rank         ();
  const idx_t  ipCount   = shape_->ipointCount ();
  const idx_t  nodeCount = shape_->nodeCount   ();

  Cubix        grads     ( rank, nodeCount, ipCount );
  Vector       weights   ( ipCount );

  Matrix       elmat     ( nodeCount, nodeCount );
  Vector       eldisp    ( nodeCount );

  Matrix       coords    ( rank, nodeCount );
  IdxVector    inodes    ( nodeCount );
  IdxVector    idofs     ( nodeCount );


  for ( idx_t ielem = 0; ielem < elems_.size(); ielem++ )
  {
    // Get the element nodes, coordinates and DOF indices.

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, jtype_ );

    // Get the element shape function gradients and integration
    // point weights.

    shape_->getShapeGradients ( grads, weights, coords );

    // Assemble the element matrix.

    elmat = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      elmat += weights[ip] * area_ * young_ *

        matmul ( grads[ip].transpose(), grads[ip] );
    }

    // Add the element matrix to the global stiffness matrix.

    mbld.addBlock ( idofs, idofs, elmat );

    // Compute the internal force vector. This is straightforward
    // because the model is linear.

    eldisp       = disp[idofs];
    fint[idofs] += matmul ( elmat, eldisp );
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

