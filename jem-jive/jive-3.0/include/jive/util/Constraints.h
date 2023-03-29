
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

#ifndef JIVE_UTIL_CONSTRAINTS_H
#define JIVE_UTIL_CONSTRAINTS_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/EventSource.h>
#include <jive/util/private/ConstraintsImp.h>


JIVE_BEGIN_PACKAGE( util )


class CConstraints;
class SConstraints;
class XConstraints;


//-----------------------------------------------------------------------
//   class Constraints
//-----------------------------------------------------------------------


class Constraints : public Object,
                    public EventSource,
                    public Clonable,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( Constraints, Object );

  enum                      Kind
  {
                              COMPRESSED,
                              SIMPLE,
                              COMPLEX
  };

  static const char*        KIND_NAMES[];

  Event < Self& >           newStructEvent;
  Event < Self& >           newRvaluesEvent;


                            Constraints     ();

  explicit                  Constraints

    ( const Ref<DofSpace>&    dofs,
      Kind                    kind = SIMPLE );

                            Constraints

    ( const Self&             rhs );

  virtual String            toString        () const override;
  virtual void              resetEvents     ()       override;

  virtual void              readFrom

    ( ObjectInput&            in )                   override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Ref<Object>       clone           () const override;

  void                      copy

    ( const Constraints&      rhs );

  void                      clear           ();

  void                      reserve

    ( idx_t                   slaveCount );

  void                      compress        ();
  void                      trimToSize      ();
  idx_t                     dofCount        () const;
  inline idx_t              slaveDofCount   () const;
  inline idx_t              masterDofCount  () const;

  inline idx_t              masterDofCount

    ( idx_t                   idof )           const;

  inline bool               isSlaveDof

    ( idx_t                   idof )           const;

  IdxVector                 getSlaveDofs    () const;

  inline idx_t              getSlaveDofs

    ( const IdxVector&        idofs )          const;

  inline bool               isMasterDof

    ( idx_t                   jdof )           const;

  inline idx_t              getMasterDofs

    ( const IdxVector&        jdofs,
      idx_t                   idof )           const;

  inline bool               hasRvalue

    ( idx_t                   idof )           const;

  inline void               addConstraint

    ( idx_t                   idof );

  inline void               addConstraint

    ( idx_t                   idof,
      double                  rval );

  inline void               addConstraint

    ( idx_t                   idof,
      idx_t                   jdof,
      double                  coeff );

  inline void               addConstraint

    ( idx_t                   idof,
      double                  rval,
      idx_t                   jdof,
      double                  coeff );

  inline void               addConstraint

    ( idx_t                   idof,
      const IdxVector&        jdofs,
      const Vector&           coeffs );

  inline void               addConstraint

    ( idx_t                   idof,
      double                  rval,
      const IdxVector&        jdofs,
      const Vector&           coeffs );

  void                      setConstraints

    ( const SparseMatrix&     conmat );

  void                      setConstraints

    ( const SparseMatrix&     conmat,
      const BoolVector&       rmask );

  bool                      eraseConstraint

    ( idx_t                   idof );

  idx_t                     eraseConstraints

    ( const IdxVector&        idofs );

  idx_t                     getConstraint

    ( double&                 rval,
      const IdxVector&        jdofs,
      const Vector&           coeffs,
      idx_t                   idof )           const;

  inline void               setRvalue

    ( idx_t                   idof,
      double                  rval );

  void                      setRvalues

    ( const IdxVector&        idofs,
      const Vector&           rvals  );

  inline void               unsetRvalue

    ( idx_t                   idof );

  inline void               unsetRvalues

    ( const IdxVector&        idofs );

  inline double             getRvalue

    ( idx_t                   idof )           const;

  void                      getRvalues

    ( const Vector&           rvals,
      const IdxVector&        idofs )          const;

  inline void               getRvalues

    ( const SparseMatrix&     cmat )           const;

  void                      getRvalueMask

    ( const BoolVector&       rmask,
      const IdxVector&        idofs )          const;

  inline SparseMatrix       toMatrix        () const;
  inline DofSpace*          getDofSpace     () const;

  void                      printTo

    ( PrintWriter&            out )            const;

  void                      store

    ( const Properties&       globdat )        const;

  static Ref<Self>          find

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );

  static Ref<Self>          extract

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );

  static Ref<Self>          get

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );


  friend void               setSlaveDofs

    ( const Vector&           vec,
      const Constraints&      cons );

  friend void               setSlaveDofs

    ( const Vector&           vec,
      const Constraints&      cons,
      double                  rscale );

  friend void               zeroSlaveDofs

    ( const Vector&           vec,
      const Constraints&      cons );

  friend void               evalSlaveDofs

    ( const Vector&           vec,
      const Constraints&      cons );

  friend void               evalSlaveDofs

    ( const Vector&           vec,
      const Constraints&      cons,
      double                  rscale );

  friend void               evalMasterDofs

    ( const Vector&           vec,
      const Constraints&      cons );


 protected:

  virtual                  ~Constraints     ();

  virtual void              emitEvents_

    ( int                     events )               override;


 private:

  void                      connect_        ();

  void                      realloc_

    ( Kind                    kind );

  void                      uncompress_     ();

  void                      uncompress_

    ( Kind                    kind );

  void                      addConstraint_

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask );

  void                      addConstraint_

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask,
      idx_t                   jdof,
      double                  coeff );

  void                      addConstraint_

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask,
      const IdxVector&        jdofs,
      const Vector&           coeffs );

  void                      dofsResized_

    ( idx_t                   count );

  void                      dofsReordered_

    ( const Reordering&       reord );


 private:

  Ref<ConstraintsImp>       cons_;

  CConstraints*             ccons_;
  SConstraints*             scons_;
  XConstraints*             xcons_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 setSlaveDofs

  ( const Vector&             vec,
    const Constraints&        cons );

inline void                 setSlaveDofs

  ( const Vector&             vec,
    const Constraints&        cons,
    double                    rscale );

inline void                 zeroSlaveDofs

  ( const Vector&             vec,
    const Constraints&        cons );

inline void                 evalSlaveDofs

  ( const Vector&             vec,
    const Constraints&        cons );

inline void                 evalSlaveDofs

  ( const Vector&             vec,
    const Constraints&        cons,
    double                    rscale );

inline void                 evalMasterDofs

  ( const Vector&             vec,
    const Constraints&        cons );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Constraints
//=======================================================================

//-----------------------------------------------------------------------
//   slaveDofCount
//-----------------------------------------------------------------------


inline idx_t Constraints::slaveDofCount () const
{
  return cons_->slaveDofCount ();
}


//-----------------------------------------------------------------------
//   masterDofCount
//-----------------------------------------------------------------------


inline idx_t Constraints::masterDofCount () const
{
  return cons_->masterDofCount ();
}


inline idx_t Constraints::masterDofCount ( idx_t idof ) const
{
  return cons_->masterDofCount ( idof );
}


//-----------------------------------------------------------------------
//   isSlaveDof
//-----------------------------------------------------------------------


inline bool Constraints::isSlaveDof ( idx_t idof ) const
{
  return cons_->isSlaveDof ( idof );
}


//-----------------------------------------------------------------------
//   isMasterDof
//-----------------------------------------------------------------------


inline bool Constraints::isMasterDof ( idx_t idof ) const
{
  return cons_->isMasterDof ( idof );
}


//-----------------------------------------------------------------------
//   getSlaveDofs
//-----------------------------------------------------------------------


inline idx_t Constraints::getSlaveDofs ( const IdxVector& idofs ) const
{
  return cons_->getSlaveDofs ( idofs );
}


//-----------------------------------------------------------------------
//   getMasterDofs
//-----------------------------------------------------------------------


inline idx_t Constraints::getMasterDofs

  ( const IdxVector&  jdofs,
    idx_t             idof ) const

{
  return cons_->getMasterDofs ( jdofs, idof );
}


//-----------------------------------------------------------------------
//   hasRvalue
//-----------------------------------------------------------------------


inline bool Constraints::hasRvalue ( idx_t idof ) const
{
  return cons_->hasRvalue ( idof );
}


//-----------------------------------------------------------------------
//   addConstraint
//-----------------------------------------------------------------------


inline void Constraints::addConstraint ( idx_t idof )
{
  addConstraint_ ( idof, 0.0, false );
}


inline void Constraints::addConstraint

  ( idx_t   idof,
    double  rval )

{
  addConstraint_ ( idof, rval, true );
}


inline void Constraints::addConstraint

  ( idx_t   idof,
    idx_t   jdof,
    double  coeff )

{
  addConstraint_ ( idof, 0.0, false, jdof, coeff );
}


inline void Constraints::addConstraint

  ( idx_t   idof,
    double  rval,
    idx_t   jdof,
    double  coeff )

{
  addConstraint_ ( idof, rval, true, jdof, coeff );
}


inline void Constraints::addConstraint

  ( idx_t             idof,
    const IdxVector&  jdofs,
    const Vector&     coeffs )

{
  addConstraint_ ( idof, 0.0, false, jdofs, coeffs );
}


inline void Constraints::addConstraint

  ( idx_t             idof,
    double            rval,
    const IdxVector&  jdofs,
    const Vector&     coeffs )

{
  addConstraint_ ( idof, rval, true, jdofs, coeffs );
}


//-----------------------------------------------------------------------
//   setRvalue
//-----------------------------------------------------------------------


inline void Constraints::setRvalue

  ( idx_t   idof,
    double  rval )

{
  cons_->setRvalue   ( idof, rval, true );
  cons_->events.emit ( this  );
}


//-----------------------------------------------------------------------
//   unsetRvalue
//-----------------------------------------------------------------------


inline void Constraints::unsetRvalue ( idx_t idof )
{
  cons_->setRvalue   ( idof, 0.0, false );
  cons_->events.emit ( this  );
}


//-----------------------------------------------------------------------
//   getRvalue
//-----------------------------------------------------------------------


double Constraints::getRvalue ( idx_t idof ) const
{
  return cons_->getRvalue ( idof );
}


//-----------------------------------------------------------------------
//   getRvalues
//-----------------------------------------------------------------------


void Constraints::getRvalues ( const SparseMatrix& cmat ) const
{
  cons_->getRvalues ( cmat );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseMatrix Constraints::toMatrix () const
{
  return cons_->toMatrix ();
}


//-----------------------------------------------------------------------
//   getDofSpace
//-----------------------------------------------------------------------


DofSpace* Constraints::getDofSpace () const
{
  return cons_->dofs_.get ();
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   setSlaveDofs
//-----------------------------------------------------------------------


inline void             setSlaveDofs

  ( const Vector&         vec,
    const Constraints&    cons )

{
  cons.cons_->setSlaveDofs ( vec, 1.0 );
}


inline void             setSlaveDofs

  ( const Vector&         vec,
    const Constraints&    cons,
    double                rscale )

{
  cons.cons_->setSlaveDofs ( vec, rscale );
}


inline void             zeroSlaveDofs

  ( const Vector&         vec,
    const Constraints&    cons )

{
  cons.cons_->setSlaveDofs ( vec, 0.0 );
}


inline void             evalSlaveDofs

  ( const Vector&         vec,
    const Constraints&    cons )

{
  cons.cons_->evalSlaveDofs ( vec, 1.0 );
}


inline void             evalSlaveDofs

  ( const Vector&         vec,
    const Constraints&    cons,
    double                rscale )

{
  cons.cons_->evalSlaveDofs ( vec, rscale );
}


inline void             evalMasterDofs

  ( const Vector&         vec,
    const Constraints&    cons )

{
  cons.cons_->evalMasterDofs ( vec );
}


JIVE_END_PACKAGE( util )

#endif
