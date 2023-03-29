
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

#ifndef JIVE_IMPLICT_STDARCLENMODEL_H
#define JIVE_IMPLICT_STDARCLENMODEL_H

#include <jem/io/Serializable.h>
#include <jive/Array.h>
#include <jive/model/Model.h>
#include <jive/implict/import.h>


JIVE_BEGIN_PACKAGE( implict )


//-----------------------------------------------------------------------
//   class StdArclenModel
//-----------------------------------------------------------------------


class StdArclenModel : public Model,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( StdArclenModel, Model );

  static const char*        TYPE_NAME;

  enum                      ArcFunc
  {
                              UNP,
                              IDC
  };


                            StdArclenModel  ();

                            StdArclenModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              readFrom

    ( ObjectInput&            in  )                  override;

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

  void                      setArcFunc

    ( ArcFunc                 func );

  inline ArcFunc            getArcFunc      () const;

  void                      setMaxIter

    ( idx_t                   count );

  inline idx_t              getMaxIter      () const;

  void                      setLoadIncr

    ( double                  incr );

  inline double             getLoadIncr     () const;

  void                      setLoadScale

    ( double                  scale );

  inline double             getLoadScale    () const;

  void                      setIncrRange

    ( double                  minIncr,
      double                  maxIncr );

  inline double             getMinIncr      () const;
  inline double             getMaxIncr      () const;

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~StdArclenModel  ();


 private:

  void                      init_           ();

  void                      init_

    ( const Properties&       globdat );

  void                      initLoad_

    ( const Properties&       globdat );

  void                      initWeights_

    ( const Properties&       globdat );

  void                      evalArcFunc_

    ( const Properties&       params,
      const Properties&       globdat );

  void                      getUnitLoad_

    ( const Properties&       params,
      const Properties&       globdat );

  void                      commit_

    ( const Properties&       params,
      const Properties&       globdat );

  void                      dofsChanged_    ();
  void                      consChanged_    ();


 private:

  static const int          U_LOAD_;
  static const int          U_WEIGHTS_;

  static const char*        DELTA_STATE_;


  Ref<Model>                child_;
  Ref<DofSpace>             dofs_;
  Ref<Constraints>          cons_;
  Ref<VectorSpace>          vspace_;

  idx_t                     istep_;
  int                       updated_;

  ArcFunc                   arcFunc_;
  idx_t                     maxIter_;
  double                    minIncr_;
  double                    maxIncr_;
  double                    loadIncr_;
  double                    loadScale_;
  double                    arcLength_;

  String                    wtblName_;

  Vector                    vtmp_;
  Vector                    load_;
  Vector                    weights_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getArcFunc
//-----------------------------------------------------------------------


inline StdArclenModel::ArcFunc

  StdArclenModel::getArcFunc () const

{
  return arcFunc_;
}


//-----------------------------------------------------------------------
//   getMaxIter
//-----------------------------------------------------------------------


inline idx_t StdArclenModel::getMaxIter () const
{
  return maxIter_;
}


//-----------------------------------------------------------------------
//   getLoadIncr
//-----------------------------------------------------------------------


inline double StdArclenModel::getLoadIncr () const
{
  return loadIncr_;
}


//-----------------------------------------------------------------------
//   getLoadScale
//-----------------------------------------------------------------------


inline double StdArclenModel::getLoadScale () const
{
  return loadScale_;
}


//-----------------------------------------------------------------------
//   get(Min|Max)Incr
//-----------------------------------------------------------------------


inline double StdArclenModel::getMinIncr () const
{
  return minIncr_;
}


inline double StdArclenModel::getMaxIncr () const
{
  return maxIncr_;
}


JIVE_END_PACKAGE( implict )

#endif
