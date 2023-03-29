
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/util/XTable.h>
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
using jive::util::XTable;
using jive::util::XDofSpace;
using jive::util::Assignable;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::algebra::MatrixBuilder;

//=======================================================================
//   class DamageModel
//=======================================================================

class DamageModel : public Model
{
 public:

                          DamageModel

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
      const Vector&         disp );

  void                    getDamage_

    ( XTable&               table );


 private:

  Assignable<NodeSet>     nodes_;
  Assignable<ElementSet>  elems_;
  Ref<XDofSpace>          dofs_;
  Ref<IShape>             shape_;
  idx_t                   jtype_;

  Matrix                  hist_;
  Matrix                  hist0_;
  Matrix                  damage_;

  double                  young_;
  double                  area_;
  double                  alpha_;
  double                  beta_;

};

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

DamageModel::DamageModel

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

  // Initialise the history parameters and the damage parameters.

  hist_  .resize ( shape_->ipointCount(), elems_.size() );
  hist0_ .resize ( shape_->ipointCount(), elems_.size() );
  damage_.resize ( shape_->ipointCount(), elems_.size() );

  hist_   = 0.0;
  hist0_  = 0.0;
  damage_ = 0.0;

  // Get the material parameters.

  myProps.get ( young_, "young" );
  myProps.get ( area_,  "area" );
  myProps.get ( alpha_, "alpha" );
  myProps.get ( beta_,  "beta" );

  // Report the used material parameters.

  myConf.set ( "young", young_ );
  myConf.set ( "area",  area_ );
  myConf.set ( "alpha", alpha_ );
  myConf.set ( "beta",  beta_ );
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

// Performs an action requested by a module or a parent model.

bool DamageModel::takeAction

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

    // TODO: store the maximum displacement and internal force
    //       as runtime variables in globdat.

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
        (table->getRowItems() == elems_.getData()) )
    {
      Vector  disp;

      StateVector::get ( disp, dofs_, globdat );

      getStress_ ( *table, disp );

      return true;
    }

    if ( (name == "damage") &&
        (table->getRowItems() == elems_.getData()) )
    {
      getDamage_ ( *table );

      return true;
    }
  }

  if ( action == Actions::COMMIT )
  {
    // Update the old history parameters.

    hist0_ = hist_;

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   assemble_
//-----------------------------------------------------------------------

// Assembles the global stiffness matrix and the internal force vector.

void DamageModel::assemble_

  ( MatrixBuilder&  mbld,
    const Vector&   fint,
    const Vector&   disp )

{
  using jem::ALL;
  using jem::max;
  using jem::numeric::matmul;

  const double  ea        = young_ * area_;

  const idx_t   rank      = nodes_.rank         ();
  const idx_t   ipCount   = shape_->ipointCount ();
  const idx_t   nodeCount = shape_->nodeCount   ();

  Cubix         grads     ( rank, nodeCount, ipCount );
  Vector        weights   ( ipCount );

  Matrix        elmat     ( nodeCount, nodeCount );
  Matrix        elmat1    ( nodeCount, nodeCount );
  Matrix        elmat2    ( nodeCount, nodeCount );
  Vector        eldisp    ( nodeCount );

  Matrix        coords    ( rank, nodeCount );
  IdxVector     inodes    ( nodeCount );
  IdxVector     idofs     ( nodeCount );

  double        strain;
  double        damage;
  double        hist;


  for ( idx_t ielem = 0; ielem < elems_.size(); ielem++ )
  {
    // Get the element nodes, coordinates and DOF indices.

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, jtype_ );

    // Get the element shape function gradients and integration
    // point weights.

    shape_->getShapeGradients ( grads, weights, coords );

    // Get the displacements in the element nodes.
    //
    eldisp = disp[idofs];

    // Assemble the linear and non-linear parts of the element matrix.

    elmat1 = 0.0;
    elmat2 = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      // Compute the strain.

      strain = dot ( grads(0,ALL,ip), eldisp );

      // TODO: compute the updated history parameter and damage
      //       parameter and store them in the members hist_ and
      //       damage_, respectively.

      // Compute the contribution to the linear part of element matrix.

      elmat1 += weights[ip] * ea * (1.0 - damage) *

        matmul ( grads[ip].transpose(), grads[ip] );

      // TODO: compute the contribution to the non-linear part
      //       of the element matrix if the damage is increasing.
      //       Store the contribution in elmat2.
    }

    // Combine the two parts of the element matrix and add the result
    // to the global stiffness matrix.

    elmat = elmat1 - elmat2;

    mbld.addBlock ( idofs, idofs, elmat );

    // TODO: update the internal force vector.
  }
}

//-----------------------------------------------------------------------
//   getStress_
//-----------------------------------------------------------------------

// Computes the average stresses in the elements.

void DamageModel::getStress_

  ( XTable&        table,
    const Vector&  disp )

{
  using jem::ALL;

  const idx_t  jcol      = table.addColumn   ( "sxx" );
  const idx_t  rank      = nodes_.rank       ();
  const idx_t  ipCount   = shape_->ipointCount ();
  const idx_t  nodeCount = shape_->nodeCount ();

  Cubix        grads     ( rank, nodeCount, ipCount );
  Matrix       coords    ( rank, nodeCount );
  IdxVector    inodes    ( nodeCount );
  IdxVector    idofs     ( nodeCount );
  Vector       eldisp    ( nodeCount );
  Vector       weights   ( ipCount );

  double       strain;
  double       stress;


  for ( idx_t ielem = 0; ielem < elems_.size(); ielem++ )
  {
    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, jtype_ );

    // Get the shape function gradients in the integration points
    // and compute the average stress.

    shape_->getShapeGradients ( grads, weights, coords );

    eldisp = disp[idofs];
    stress = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      strain  = dot ( grads(0,ALL,ip), eldisp );
      stress += weights[ip] * young_ *
                (1.0 - damage_(ip,ielem)) * strain;
    }

    stress /= sum ( weights );

    // Store the average stress in the table.

    table.setValue ( ielem, jcol, stress );
  }
}

//-----------------------------------------------------------------------
//   getDamage_
//-----------------------------------------------------------------------

// Returns the average damage in the elements.

void DamageModel::getDamage_ ( XTable& table )
{
  using jem::ALL;

  const idx_t  jcol      = table.addColumn     ( "d" );
  const idx_t  rank      = nodes_.rank         ();
  const idx_t  ipCount   = shape_->ipointCount ();
  const idx_t  nodeCount = shape_->nodeCount   ();

  Matrix       coords    ( rank, nodeCount );
  IdxVector    inodes    ( nodeCount );
  Vector       weights   ( ipCount );

  double       damage;


  for ( idx_t ielem = 0; ielem < elems_.size(); ielem++ )
  {
    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );

    shape_->getIntegrationWeights ( weights, coords );

    damage  = dot ( weights, damage_[ielem] );
    damage /= sum ( weights );

    table.setValue ( ielem, jcol, damage );
  }
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newDamageModel
//-----------------------------------------------------------------------

// Creates a new DamageModel instance.

Ref<Model>           newDamageModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::newInstance;

  return newInstance<DamageModel> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareDamageModel
//-----------------------------------------------------------------------

// Registers the DamageModel class with the ModelFactory.

void declareDamageModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "Damage", newDamageModel );
}

