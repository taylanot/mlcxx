
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/assert.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/tuple/utilities.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/io/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/declare.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/DiagMatrixExtension.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/algebra/MBuilderParams.h>
#include <jive/model/Names.h>
#include <jive/model/Actions.h>
#include <jive/model/ActionException.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/MatrixModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::MatrixModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;
using jem::System;
using jem::util::mergeAndReplace;
using jive::util::joinNames;


//=======================================================================
//   class MatrixModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MatrixModel::TYPE_NAME = "Matrix";

const char*  MatrixModel::MATRIX_NAMES[3] =
{
  PropNames::MATRIX0,
  PropNames::MATRIX1,
  PropNames::MATRIX2
};


//-----------------------------------------------------------------------
//   constructors & destructors
//-----------------------------------------------------------------------


MatrixModel::MatrixModel ()
{}


MatrixModel::MatrixModel

  ( const String&         name,
    const Ref<DofSpace>&  dofs,
    const Ref<Model>&     child ) :

    Super  ( name  ),
    dofs_  ( dofs  ),
    child_ ( child )

{
  JEM_PRECHECK ( dofs && child );

  init_    ();
  connect_ ();
}


MatrixModel::MatrixModel

  ( const String&         name,
    const Properties&     conf,
    const Properties&     props,
    const Ref<DofSpace>&  dofs,
    const Ref<Model>&     child ) :

    Super  ( name  ),
    dofs_  ( dofs  ),
    child_ ( child )

{
  JEM_PRECHECK ( dofs && child );

  init_    ();
  connect_ ();

  myProps_ = props.findProps ( myName_ );
  myConf_  = conf .makeProps ( myName_ );
}


MatrixModel::MatrixModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name )

{
  using jive::algebra::declareMBuilders;

  init_ ();

  // First create the child model and then get the DOF space.

  child_ = newModel ( joinNames ( myName_, PropNames::MODEL ),
                      conf,
                      props,
                      globdat );

  dofs_    = DofSpace::get ( globdat, getContext() );
  myProps_ = props.findProps ( myName_ );
  myConf_  = conf .makeProps ( myName_ );

  myProps_.find ( debug_,  PropNames::DEBUG  );
  myProps_.find ( linear_, PropNames::LINEAR );

  myConf_ .set  ( PropNames::DEBUG,  debug_  );
  myConf_ .set  ( PropNames::LINEAR, linear_ );

  declareMBuilders ();
  connect_         ();
}


MatrixModel::~MatrixModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void MatrixModel::readFrom ( ObjectInput& in )
{
  // Clear the current state.

  for ( int i = 0; i < 3; i++ )
  {
    mdata_[i].mbuilder = nullptr;
  }

  init_  ();

  decode ( in,
           myName_,
           dofs_,
           child_ );

  decode ( in,
           debug_,
           linear_,
           myProps_,
           myConf_ );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void MatrixModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           dofs_,
           child_ );

  encode ( out,
           debug_,
           linear_,
           myProps_,
           myConf_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* MatrixModel::findModel ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    return child_->findModel ( name );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MatrixModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    mergeAndReplace ( myProps_, props.findProps( myName_ ) );

    myProps_.find ( debug_,  PropNames::DEBUG  );
    myProps_.find ( linear_, PropNames::LINEAR );

    for ( int i = 0; i < 3; i++ )
    {
      MData_&  md = mdata_[i];

      if ( ! md.mbuilder )
      {
        continue;
      }

      md.mbuilder->configure ( props );

      if ( myProps_.contains( MATRIX_NAMES[i] ) )
      {
        Properties  matProps =

          myProps_.findProps ( MATRIX_NAMES[i] );

        bool        isConst  = md.isConst;

        matProps.find ( isConst, PropNames::CONSTANT );

        if ( md.isConst && ! isConst )
        {
          md.updated = false;
        }

        md.isConst = isConst;
      }
    }
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MatrixModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  myConf_.set ( PropNames::DEBUG,  debug_  );
  myConf_.set ( PropNames::LINEAR, linear_ );

  mergeAndReplace ( conf.makeProps( myName_ ), myConf_ );

  for ( int i = 0; i < 3; i++ )
  {
    const MData_&  md = mdata_[i];

    if ( md.mbuilder )
    {
      Properties  matConf = myConf_.makeProps ( MATRIX_NAMES[i] );

      md.mbuilder->getConfig ( conf );

      matConf.set ( PropNames::CONSTANT, md.isConst );
    }
  }

  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool MatrixModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result;


  if ( action == Actions::GET_INT_VECTOR )
  {
    return getIntVector_ ( params, globdat );
  }

  for ( int i = 0; i < 3; i++ )
  {
    if ( action == Actions::UPD_MATRIX[i] )
    {
      MData_&  md = mdata_[i];

      if ( ! md.mbuilder )
      {
        // Let the child model handle this action.

        return child_->takeAction ( action, params, globdat );
      }

      if ( md.updated && md.isConst )
      {
        if ( debug_ > 0 )
        {
          print ( System::debug( myName_ ), myName_,
                  " : ", ActionParams::MATRIX[i],
                  " is constant; no update needed\n" );
        }

        if ( i == 0 )
        {
          // Update the internal force vector.

          return getIntVector_ ( params, globdat );
        }
        else
        {
          return true;
        }
      }

      if ( debug_ > 0 )
      {
        print ( System::debug( myName_ ), myName_,
                " : updating ", ActionParams::MATRIX[i],
                " ...\n" );
      }

      md.mbuilder->setToZero ();

      params.set ( ActionParams::MATRIX[i], md.mbuilder );

      result = child_->takeAction ( Actions::GET_MATRIX[i],
                                    params, globdat );

      params.erase ( ActionParams::MATRIX[i] );

      try
      {
        md.mbuilder->updateMatrix ();
      }
      catch ( jem::Exception& ex )
      {
        ex.setContext ( getContext() );
        throw;
      }

      md.updated = true;

      // Make sure to be notified if the DofSpace size changes.

      dofs_->resetEvents ();

      if ( result && debug_ > 0 )
      {
        print ( System::debug( myName_ ), myName_,
                " : updated ", ActionParams::MATRIX[i], "\n" );

        if      ( debug_ > 2 )
        {
          printMatrix_  ( i );
        }
        else if ( debug_ > 1 )
        {
          printSummary_ ( i );
        }
      }

      return result;
    }
  }

  for ( int i = 0; i < 3; i++ )
  {
    if ( action == Actions::NEW_MATRIX[i] )
    {
      // Give the child model the chance to create its own
      // matrix. Note that the return value of the child's takeAction
      // method is ignored; this avoids obscure errors when the return
      // value is somehow incorrect.

      params.erase ( ActionParams::MATRIX[i] );

      child_->takeAction ( action, params, globdat );

      if ( params.contains( ActionParams::MATRIX[i] ) )
      {
        return true;
      }

      Ref<AbstractMatrix>  mat;

      try
      {
        mat = newMatrix ( i, globdat );
      }
      catch ( const jem::Exception& ex )
      {
        throw ActionException (
          myName_,
          action,
          "failed to create a new matrix: " + ex.what()
        );
      }

      if ( debug_ > 0 )
      {
        print ( System::debug( myName_ ), myName_,
                " : created ", ActionParams::MATRIX[i],
                " : ", mat->toString (), "\n\n" );
      }

      params.set ( ActionParams::MATRIX[i], mat );

      return true;
    }
  }

  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   newMatrix
//-----------------------------------------------------------------------


Ref<AbstractMatrix>  MatrixModel::newMatrix

  ( int                imat,
    const Properties&  globdat )

{
  using jive::algebra::newMBuilder;
  using jive::algebra::MBuilderParams;

  JEM_PRECHECK ( imat >= 0 && imat < 3 );

  MData_&  md = mdata_[imat];

  if ( ! md.mbuilder )
  {
    const String  name   =

      joinNames ( myName_, MATRIX_NAMES[imat] );

    Properties    params =

      MBuilderParams::newInstance ( dofs_ );

    Properties    props, conf;


    // All higher order matrices are assumed to be constant, unless
    // specified otherwise.

    md.isConst = (imat != 0);
    md.updated = false;

    if ( myProps_.contains( PropNames::MATRIX ) )
    {
      // Set the default properties. To avoid warnings about
      // unused properties the function mergeNoReplace is not
      // used.

      Properties    matProps  =

        myProps_.getProps  ( PropNames::MATRIX );

      Properties    imatProps =

        myProps_.makeProps ( MATRIX_NAMES[imat] );

      StringVector  propList  = matProps.listProps ();

      Ref<Object>   obj;

      for ( int i = 0; i < propList.size(); i++ )
      {
        if ( ! imatProps.contains( propList[i] ) )
        {
          matProps .get ( obj, propList[i] );
          imatProps.set ( propList[i], obj );
        }
      }
    }

    if ( myProps_.contains( MATRIX_NAMES[imat] ) )
    {
      Properties  matProps =

        myProps_.findProps ( MATRIX_NAMES[imat] );

      matProps.find ( md.isConst, PropNames::CONSTANT );
    }

    props.set ( myName_, myProps_ );
    conf .set ( myName_, myConf_  );

    md.mbuilder = newMBuilder ( name, conf, props, params, globdat );

    if ( ! md.mbuilder )
    {
      throw jem::IllegalInputException (
        getContext (),
        "failed to create a matrix builder "
        "(invalid/conflicting application data)"
      );
    }

    md.mbuilder->configure ( props );
    md.mbuilder->getConfig ( conf  );
  }

  return md.mbuilder->getMatrix ();
}


//-----------------------------------------------------------------------
//   setLinear
//-----------------------------------------------------------------------


void MatrixModel::setLinear ( bool yesno )
{
  linear_ = yesno;
}


//-----------------------------------------------------------------------
//   setDebugLevel
//-----------------------------------------------------------------------


void MatrixModel::setDebugLevel ( int level )
{
  debug_ = level;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> MatrixModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MatrixModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & Self::makeNew );
  ModelFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void MatrixModel::init_ ()
{
  debug_  = 0;
  linear_ = false;

  for ( int i = 0; i < 3; i++ )
  {
    MData_&  md = mdata_[i];

    md.isConst = false;
    md.updated = false;
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void MatrixModel::connect_ ()
{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent, this, & Self::dofsChanged_ );

  dofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   getIntVector_
//-----------------------------------------------------------------------


bool MatrixModel::getIntVector_

  ( const Properties&  params,
    const Properties&  globdat )

{
  MData_&  md = mdata_[0];

  // Take a shortcut if possible.

  if ( linear_ && md.updated && md.mbuilder )
  {
    Ref<AbstractMatrix>    m0 = md.mbuilder->getMatrix ();
    AbstractMatrix::Shape  sh = m0->shape ();

    // Make sure that the matrix has the correct shape.

    if ( sh[0] == sh[1] && sh[0] == dofs_->dofCount() )
    {
      Vector  t ( sh[0] );
      Vector  u, f;

      StateVector::get ( u, dofs_, globdat );
      params      .get ( f, ActionParams::INT_VECTOR );

      m0->matmul ( t, u );

      f += t;

      return true;
    }
  }

  // No shortcut possible, so ask the child model for the internal
  // force vector.

  return child_->takeAction ( Actions::GET_INT_VECTOR,
                              params, globdat );
}


//-----------------------------------------------------------------------
//   printMatrix_
//-----------------------------------------------------------------------


void MatrixModel::printMatrix_ ( int imat )
{
  using jem::io::getWidthOf;
  using jem::io::Writer;
  using jem::io::PrintWriter;

  Ref<AbstractMatrix>  mat = mdata_[imat].mbuilder->getMatrix ();

  const int            iw  = getWidthOf ( max( mat->shape() ) );

  Ref<PrintWriter>     prn =

    newInstance<PrintWriter> ( & System::debug( myName_ ) );

  prn->nformat.setScientific     ();
  prn->nformat.setFloatWidth     ( 16 );
  prn->nformat.setFractionDigits (  8 );
  prn->nformat.setIntegerWidth   ( iw );

  prn->incrIndentLevel ();
  prn->printLine       ();

  mdata_[imat].mbuilder->getMatrix()->printTo ( *prn );

  prn->printBlankLine  ();
  prn->flush           ();
}


//-----------------------------------------------------------------------
//   printSummary_
//-----------------------------------------------------------------------


void MatrixModel::printSummary_ ( int imat )
{
  using jem::min;
  using jem::max;
  using jem::io::endl;
  using jem::io::Writer;
  using jive::algebra::DiagMatrixExt;

  Writer&  debug = System::debug ( myName_ );

  Ref<AbstractMatrix>  mat = mdata_[imat].mbuilder->getMatrix ();
  DiagMatrixExt*       dia = mat->getExtension<DiagMatrixExt> ();

  print ( debug, myName_, " : matrix shape : ",
          mat->shape(), endl );

  if ( dia )
  {
    Vector  s ( mat->size(0) );

    dia->getRowScales ( s );

    print ( debug, myName_, " : max absolute row value : ",
            max( s ), endl );

    print ( debug, myName_, " : min absolute row value : ",
            min( s ), endl );
  }
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void MatrixModel::dofsChanged_ ()
{
  for ( int i = 0; i < 3; i++ )
  {
    mdata_[i].updated = false;
  }

  if ( debug_ > 0 )
  {
    print ( System::debug( myName_ ), myName_,
            " : number of DOFs changed; invalidated matrices\n" );
  }
}


JIVE_END_PACKAGE( model )
