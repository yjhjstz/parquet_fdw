#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"

#include "arrow_bytea.h"


#define PG_GETARG_ARROWBYTEA(i) (arrow_bytea *) PG_GETARG_POINTER(i)

PG_FUNCTION_INFO_V1(arrow_bytea_in);
PG_FUNCTION_INFO_V1(arrow_bytea_out);
PG_FUNCTION_INFO_V1(arrow_bytea2bytea);
PG_FUNCTION_INFO_V1(arrow_bytea_cmp);
PG_FUNCTION_INFO_V1(arrow_bytea_lt);
PG_FUNCTION_INFO_V1(arrow_bytea_gt);
PG_FUNCTION_INFO_V1(arrow_bytea_le);
PG_FUNCTION_INFO_V1(arrow_bytea_ge);
PG_FUNCTION_INFO_V1(arrow_bytea_eq);

arrow_bytea *
make_arrow_bytea(const char *data, int64_t len)
{
    arrow_bytea *ret = (arrow_bytea *) palloc0(sizeof(arrow_bytea));

    ret->len = len;
    ret->data = data;

    return ret;
}

Datum
arrow_bytea_in(PG_FUNCTION_ARGS)
{
    elog(ERROR, "input function for arrow_bytea is not implemented");
}

Datum
arrow_bytea_out(PG_FUNCTION_ARGS)
{
    elog(ERROR, "output function for arrow_bytea is not implemented");
}

Datum
arrow_bytea2bytea(PG_FUNCTION_ARGS)
{
    arrow_bytea *ab = PG_GETARG_ARROWBYTEA(0);

    PG_RETURN_BYTEA_P(cstring_to_text_with_len(ab->data, ab->len));
}

static int
arrow_bytea_cmp_internal(arrow_bytea *a, arrow_bytea *b)
{
    return memcmp(a, b, Min(a->len, b->len));
}

Datum
arrow_bytea_cmp(PG_FUNCTION_ARGS)
{
    arrow_bytea *a = PG_GETARG_ARROWBYTEA(0);
    arrow_bytea *b = PG_GETARG_ARROWBYTEA(1);
    
    PG_RETURN_INT32(arrow_bytea_cmp_internal(a, b));
}

Datum
arrow_bytea_lt(PG_FUNCTION_ARGS)
{
    arrow_bytea *a = PG_GETARG_ARROWBYTEA(0);
    arrow_bytea *b = PG_GETARG_ARROWBYTEA(1);
    
    PG_RETURN_BOOL(arrow_bytea_cmp_internal(a, b) < 0);
}

Datum
arrow_bytea_gt(PG_FUNCTION_ARGS)
{
    arrow_bytea *a = PG_GETARG_ARROWBYTEA(0);
    arrow_bytea *b = PG_GETARG_ARROWBYTEA(1);
    
    PG_RETURN_BOOL(arrow_bytea_cmp_internal(a, b) > 0);
}

Datum
arrow_bytea_le(PG_FUNCTION_ARGS)
{
    arrow_bytea *a = PG_GETARG_ARROWBYTEA(0);
    arrow_bytea *b = PG_GETARG_ARROWBYTEA(1);

    PG_RETURN_BOOL(arrow_bytea_cmp_internal(a, b) <= 0);
}

Datum
arrow_bytea_ge(PG_FUNCTION_ARGS)
{
    arrow_bytea *a = PG_GETARG_ARROWBYTEA(0);
    arrow_bytea *b = PG_GETARG_ARROWBYTEA(1);

    PG_RETURN_BOOL(arrow_bytea_cmp_internal(a, b) >= 0);
}

Datum
arrow_bytea_eq(PG_FUNCTION_ARGS)
{
    arrow_bytea *a = PG_GETARG_ARROWBYTEA(0);
    arrow_bytea *b = PG_GETARG_ARROWBYTEA(1);

    PG_RETURN_BOOL(arrow_bytea_cmp_internal(a, b) == 0);
}
