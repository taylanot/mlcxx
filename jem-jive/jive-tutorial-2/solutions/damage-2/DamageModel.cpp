
#include <jem/base/Array.h>
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

  Matrix                  hist_;
  Matrix                  hist0_;
  Matrix                  damage_;

  double                  young_;
  double                  poisson_;
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

  // Initialise the history parameters and the damage parameters.

  hist_  .resize ( shape_->ipointCount(), group_.size() );
  hist0_ .resize ( shape_->ipointCount(), group_.size() );
  damage_.resize ( shape_->ipointCount(), group_.size() );

  hist_   = 0.0;
  hist0_  = 0.0;
  damage_ = 0.0;

  // Get the material parameters.

  myProps.get ( young_,   "young" );
  myProps.get ( poisson_, "poisson" );
  myProps.get ( alpha_,   "alpha" );
  myProps.get ( beta_,    "beta" );

  // Report the used material parameters.

  myConf.set ( "young",   young_ );
  myConf.set ( "poisson", poisson_ );
  myConf.set ( "alpha",   alpha_ );
  myConf.set ( "beta",    beta_ );
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

    // Store the maximum displacement and internal force.

    globdat.set ( "var.load", max( abs( fint ) ) );
    globdat.set ( "var.disp", max( abs( disp ) ) );

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
  Matrix        elmat1    ( dofCount, dofCount );
  Matrix        elmat2    ( dofCount, dofCount );
  Vector        eldisp    ( dofCount );

  Matrix        coords    ( rank, nodeCount );
  IdxVector     inodes    ( nodeCount );
  IdxVector     idofs     ( dofCount );

  Matrix        strmat    ( 3, 3 );
  Vector        strain    ( 3 );
  Vector        stress    ( 3 );

  MatmulChain
    <double,3>  mc3;

  double        damexp;
  double        damage;
  double        dilat;
  double        deriv;
  double        hist;


  // Get the linear strain-stress stiffness matrix.

  getStiffmat_ ( stiff );

  strmat = 0.0;

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

    // Get the displacements in the element nodes.

    eldisp = disp[idofs];

    // Assemble the linear and non-linear parts of the element matrix.

    elmat1 = 0.0;
    elmat2 = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      getBMatrix_ ( bmat, grads[ip] );

      // Compute the strain, stress and the history parameter.

      matmul ( strain, bmat,  eldisp );
      matmul ( stress, stiff, strain );

      dilat  = strain[0] + strain[1];
      hist   = jem::max ( hist0_(ip,ie), dilat );

      // Compute the damage parameter.

      if ( hist <= alpha_ )
      {
        damage = 0.0;
      }
      else
      {
        // Exponential damage formulation.

        damexp = std::exp ( -(hist - alpha_) / (beta_ - alpha_) );
        damage = 1.0 - alpha_ / hist *  damexp;
        damage = jem::max ( damage, 0.0 );
        damage = jem::min ( damage, 1.0 );
      }

      // Store the updated history and damage parameters.

      hist_  (ip,ie) = hist;
      damage_(ip,ie) = damage;

      // Compute the contribution to the linear part of element matrix.

      elmat1 += weights[ip] * (1.0 - damage) *

        mc3.matmul ( bmat.transpose(), stiff, bmat );

      if ( (hist > hist0_(ip,ie)) && (hist > alpha_) )
      {
        // Damage increases; compute the contribution to the non-linear
        // part of the element matrix.

        damexp    = std::exp ( -(hist - alpha_) / (beta_ - alpha_) );
        deriv     = -(1.0 / hist + 1.0 / (beta_ - alpha_));
        deriv    *= alpha_ * damexp / hist;

        strmat[0] = stress;
        strmat[1] = stress;
        elmat2   += weights[ip] * deriv *
                    mc3.matmul ( bmat.transpose(), strmat, bmat );
      }
    }

    // Combine the two parts of the element matrix and add the result
    // to the global stiffness matrix.

    elmat = elmat1 - elmat2;

    mbld.addBlock ( idofs, idofs, elmat );

    // Compute the internal force vector.

    fint[idofs] += matmul ( elmat1, eldisp );
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
  using jem::numeric::matmul;

  const idx_t  rank      = nodes_.rank         ();
  const idx_t  ipCount   = shape_->ipointCount ();
  const idx_t  nodeCount = shape_->nodeCount   ();
  const idx_t  dofCount  = rank * nodeCount;

  IdxVector    ielems    = group_.getIndices   ();

  Cubix        grads     ( rank, nodeCount, ipCount );
  Matrix       bmat      ( 3,    dofCount );
  Matrix       stiff     ( 3,    3 );
  Matrix       coords    ( rank, nodeCount );
  Vector       eldisp    ( dofCount );
  Vector       weights   ( ipCount );
  IdxVector    inodes    ( nodeCount );
  IdxVector    idofs     ( dofCount );
  IdxVector    jcols     ( 3 );
  Vector       strain    ( 3 );
  Vector       stress    ( 3 );


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

    // Get the shape function gradients in the integration points
    // and compute the average stress.

    shape_->getShapeGradients ( grads, weights, coords );

    eldisp = disp[idofs];
    stress = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      getBMatrix_ ( bmat, grads[ip] );

      matmul ( strain, bmat, eldisp );

      stress += weights[ip] * (1.0 - damage_(ip,ie)) *

        matmul ( stiff, strain );
    }

    stress /= sum ( weights );

    // Store the average stress in the table.

    table.setRowValues ( ielem, jcols, stress );
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

  IdxVector    ielems    = group_.getIndices   ();

  Matrix       coords    ( rank, nodeCount );
  IdxVector    inodes    ( nodeCount );
  Vector       weights   ( ipCount );

  double       damage;


  for ( idx_t ie = 0; ie < ielems.size(); ie++ )
  {
    idx_t  ielem = ielems[ie];

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords, inodes );

    shape_->getIntegrationWeights ( weights, coords );

    damage  = dot ( weights, damage_[ie] );
    damage /= sum ( weights );

    table.setValue ( ielem, jcol, damage );
  }
}

//-----------------------------------------------------------------------
//   getStiffmat_
//-----------------------------------------------------------------------

void DamageModel::getStiffmat_ ( const Matrix& stiff ) const
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


void DamageModel::getBMatrix_

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

