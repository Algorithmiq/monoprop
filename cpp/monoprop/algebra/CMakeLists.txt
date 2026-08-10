target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Algebra.h"
        "AlgebraCommon.h"
        "MajoranaAlgebra.h"
        "PauliAlgebra.h"
)
