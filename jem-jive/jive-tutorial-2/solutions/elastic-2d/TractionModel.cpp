
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Assignable.h>
#include <jive/geom/Line.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include "import.h"
#include "declare.h"
#include "Array.h"

using jive::geom::Shape;
using jive::model::Model;
using jive::util::DofSpace;
using jive::util::Assignable;
using jive::fem::NodeSet;
using jive::fem::ElementSet;

//=======================================================================
//   class TractionModel
//=======================================================================

class TractionModel : public Model
{
 public:

                          TractionModel

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

    ( const Vector&         fext );


 private:

  Assignable<NodeSet>     nodes_;
  Assignable<ElementSet>  elems_;
  Ref<DofSpace>           dofs_;
  Ref<Shape>              shape_;
  idx_t                   jtype_;

  double                  tract_;

};

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

TractionModel::TractionModel

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

  // Get the DofSpace.

  dofs_  = DofSpace::get ( nodes_.getData(), globdat, getContext() );
  jtype_ = dofs_->getTypeIndex ( "u" );

  // Get the model parameters.

  myProps.get ( tract_, "tract" );

  // Report the used material parameters.

  myConf.set ( "tract", tract_ );
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

// Performs an action requested by a module or a parent model.

bool TractionModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

  if ( action == Actions::GET_EXT_VECTOR )
  {
    Vector  fext;

    // Get the action-specific parameters.

    params.get ( fext, ActionParams::EXT_VECTOR );

    // Assemble the external vector.

    assemble_ ( fext );

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   assemble_
//-----------------------------------------------------------------------

// Assembles the external force vector.

void TractionModel::assemble_ ( const Vector& fext )
{
  const idx_t  rank      = nodes_.rank         ();
  const idx_t  ipCount   = shape_->ipointCount ();
  const idx_t  nodeCount = shape_->nodeCount   ();

  Matrix       sfuncs    ( nodeCount, ipCount );
  Vector       weights   ( ipCount );

  Vector       elvec     ( nodeCount );

  Matrix       coords    ( rank, nodeCount );
  IdxVector    inodes    ( nodeCount );
  IdxVector    idofs     ( nodeCount );


  // Get the shape fucntions in the integration points. These values
  // are the same for all elements.

  sfuncs = shape_->getShapeFunctions ();

  for ( idx_t ielem = 0; ielem < elems_.size(); ielem++ )
  {
    // Get the element nodes, coordinates and DOF indices.

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, jtype_ );

    // Get the integration point weights.

    shape_->getIntegrationWeights ( weights, coords );

    // Assemble the element vector.

    elvec = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      elvec += weights[ip] * tract_ * sfuncs[ip];
    }

    // Add the element vector to the global vector.

    fext[idofs] += elvec;
  }
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newTractionModel
//-----------------------------------------------------------------------

// Creates a new TractionModel instance.

Ref<Model>           newTractionModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::newInstance;

  return newInstance<TractionModel> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareTractionModel
//-----------------------------------------------------------------------

// Registers the TractionModel class with the ModelFactory.

void declareTractionModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "Traction", newTractionModel );
}

