/*
 *  StressModel Header 
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */

#ifndef TEST_MODEL_H 
#define TEST_MODEL_H

#include <jive/model/Model.h>
#include <jem/base/Thread.h>
#include <jem/base/Monitor.h>
#include <jem/numeric/func/Function.h>
#include <jive/util/XTable.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Assignable.h>
#include <jive/util/utilities.h>
#include <jive/util/StdPointSet.h>
#include <jive/geom/InternalShape.h>
#include <jive/fem/ElementGroup.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jem/io/Writer.h>
#include <jem/io/PrintWriter.h>

//#include "../materials/Material.h"
//#include "DispWriter.h"


using namespace jem;

using jem::util::Properties;
using jem::numeric::Function;
using jive::Vector;
using jive::IdxVector;
using jive::Matrix;
using jive::Cubix;
using jive::util::XTable;
using jive::util::XDofSpace;
using jive::util::Assignable;
using jive::util::XPointSet;
using jive::model::Model;
using jive::geom::IShape;
using jive::fem::ElementGroup;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::algebra::MatrixBuilder;

using jem::io::Writer;
using jem::io::PrintWriter;


//------------------------------------------------------------------------------
//   class StressModel
//------------------------------------------------------------------------------

class StressModel : public Model
{
  public:

  StressModel
    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );


  virtual bool takeAction
    ( const String&     action,
      const Properties& params,
      const Properties& globdat );

  static Ref<Model> makeNew
    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  static const char* TYPE_NAME;
  static const char* DOF_TYPE_NAMES[3];
  static const char* SHAPE_PROP;
  static const char* MATERIAL_PROP;
  static const char* THICKNESS_PROP;
  static const char* WRITESTATE_PROP;
  static const char* WRITESNAPS_PROP;
  static const char* WRITESTRAINS_PROP;
  static const char* WRITESTRESSES_PROP;
  static const char* WRITEIPCOORDS_PROP;

 private:

  void initDofs_
    ( const Properties& globdat );

  void getMatrix_
    ( const Vector&      fint,
      Ref<MatrixBuilder> mbld,
      const Properties&  globdat );

  void getDissForce_
    ( const Vector& fDiss,
      const Vector& disp );

  bool getTable_
    ( const Properties& params,
      const Properties& globdat );

  Ref<PrintWriter> initWriter_
    ( const Properties& params,
      const String      name ) const;

  void printIps_
    ( const Properties& params,
      const Properties& globdat );

  void printState_
    ( const Properties& params,
      const Properties& globdat );

  void printStrains_
    ( const Properties& params,
      const Properties& globdat );

  void printStresses_
    ( const Properties& params,
      const Properties& globdat );

  void printNodalStresses_
    ( XTable&           table,
      const String&     tablename,
      const Vector&     weights,
      const Properties& globdat );

  void printNodalHistory_
    ( XTable&           table,
      const String&     tablename,
      const Vector&     weights,
      const Properties& globdat );

  void printFullDofs_
    ( XTable&           table,
      const String&     tablename,
      const Vector&     weights,
      const Properties& globdat );

  void getShapeGrads_
    ( const Matrix& b,
      const Matrix& g );

  void get1DShapeGrads_
    ( const Matrix& b,
      const Matrix& g );

  void get2DShapeGrads_
    ( const Matrix& b,
      const Matrix& g );

  void get3DShapeGrads_
    ( const Matrix& b,
      const Matrix& g );

  void getDissipation_ 
    ( const Properties& params ) const;

  private:


  Properties props_;

  Assignable<ElementGroup> egroup_;
  Assignable<ElementSet>   elems_;
  Assignable<NodeSet>      nodes_;

  idx_t rank_;
  idx_t nodeCount_;
  idx_t numElem_;
  idx_t ipCount_;

  Ref<IShape> shape_;
  //Ref<Material>             material_;

  Ref<XDofSpace> dofs_;
  IdxVector      dofTypes_;
  idx_t          dofCount_;

  idx_t strCount_;

  double        thickness_;
  Ref<Function> thickFunc_;

  //DispWriter                dw_;

  // Writers

  Ref<PrintWriter> stateOut_;
  Ref<PrintWriter> strainOut_;
  Ref<PrintWriter> stressOut_;
  Ref<PrintWriter> coordOut_;

  // DispWriter                stateWriter_;
  // DispWriter                epsWriter_;
  // DispWriter                sigWriter_;

  bool writeState_;
  bool writeStrains_;
  bool writeStresses_;

  Matrix epss_;
  Matrix sigs_;
  Cubix  stiffs_;
  Matrix ipcoords_;


};

//------------------------------------------------------------------------------
//    declareStressModel
//------------------------------------------------------------------------------

void declareStressModel ();

#endif
