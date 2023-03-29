
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

#ifndef JIVE_MODEL_MATRIXMODEL_H
#define JIVE_MODEL_MATRIXMODEL_H

#include <jem/io/Serializable.h>
#include <jem/util/Properties.h>
#include <jive/model/import.h>
#include <jive/model/Model.h>


JIVE_BEGIN_PACKAGE( model )


//-----------------------------------------------------------------------
//   class MatrixModel
//-----------------------------------------------------------------------


class MatrixModel : public Model,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( MatrixModel, Model );

  static const char*        TYPE_NAME;
  static const char*        MATRIX_NAMES[3];


                            MatrixModel     ();

                            MatrixModel

    ( const String&           name,
      const Ref<DofSpace>&    dofs,
      const Ref<Model>&       child );

                            MatrixModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Ref<DofSpace>&    dofs,
      const Ref<Model>&       child );

                            MatrixModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              readFrom

    ( ObjectInput&            in )                   override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Model*            findModel

    ( const String&           name )           const override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )              override;

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )        const override;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat )              override;

  Ref<AbstractMatrix>       newMatrix

    ( int                     imat,
      const Properties&       globdat );

  void                      setLinear

    ( bool                    yesno = true );

  inline bool               isLinear        () const;

  void                      setDebugLevel

    ( int                     level );

  inline int               getDebugLevel   () const;

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~MatrixModel     ();


 private:

  void                      init_           ();
  void                      connect_        ();

  void                      printMatrix_

    ( int                     imat );

  void                      printSummary_

    ( int                     imat );

  bool                      getIntVector_

    ( const Properties&       params,
      const Properties&       globdat );

  void                      dofsChanged_    ();


 private:

  struct                    MData_
  {
    Ref<MatrixBuilder>        mbuilder;

    bool                      isConst;
    bool                      updated;
  };


  Ref<DofSpace>             dofs_;
  Ref<Model>                child_;

  int                       debug_;
  bool                      linear_;

  Properties                myConf_;
  Properties                myProps_;

  MData_                    mdata_[3];

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


inline bool MatrixModel::isLinear () const
{
  return linear_;
}


//-----------------------------------------------------------------------
//   getDebugLevel
//-----------------------------------------------------------------------


inline int MatrixModel::getDebugLevel () const
{
  return debug_;
}


JIVE_END_PACKAGE( model )

#endif
