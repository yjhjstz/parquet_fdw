-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION parquet_fdw" to load this file. \quit

CREATE FUNCTION parquet_fdw_handler()
RETURNS fdw_handler
AS 'MODULE_PATHNAME'
LANGUAGE C STRICT;

CREATE FUNCTION parquet_fdw_validator(text[], oid)
RETURNS void
AS 'MODULE_PATHNAME'
LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER parquet_fdw
  HANDLER parquet_fdw_handler
  VALIDATOR parquet_fdw_validator;

CREATE TYPE arrow_bytea;

CREATE FUNCTION arrow_bytea_in (cstring) RETURNS arrow_bytea
AS 'parquet_fdw', 'arrow_bytea_in'
LANGUAGE C STRICT IMMUTABLE;

CREATE FUNCTION arrow_bytea_out (arrow_bytea) RETURNS cstring
AS 'parquet_fdw', 'arrow_bytea_out'
LANGUAGE C STRICT IMMUTABLE;


CREATE TYPE arrow_bytea
(
    input = arrow_bytea_in,
    output = arrow_bytea_out,
    internallength = 16
);

CREATE FUNCTION arrow_bytea2bytea(arrow_bytea) RETURNS bytea
AS 'parquet_fdw', 'arrow_bytea2bytea'
LANGUAGE C STRICT;

CREATE FUNCTION arrow_bytea2text(arrow_bytea) RETURNS text
AS 'parquet_fdw', 'arrow_bytea2bytea'
LANGUAGE C STRICT;

CREATE CAST (arrow_bytea AS bytea)
WITH FUNCTION arrow_bytea2bytea;

CREATE CAST (arrow_bytea AS text)
WITH FUNCTION arrow_bytea2text;

CREATE FUNCTION arrow_bytea_cmp(arrow_bytea, arrow_bytea) RETURNS int4
AS 'parquet_fdw', 'arrow_bytea_cmp'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION arrow_bytea_lt(arrow_bytea, arrow_bytea) RETURNS bool
AS 'parquet_fdw', 'arrow_bytea_lt'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION arrow_bytea_gt(arrow_bytea, arrow_bytea) RETURNS bool
AS 'parquet_fdw', 'arrow_bytea_gt'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION arrow_bytea_le(arrow_bytea, arrow_bytea) RETURNS bool
AS 'parquet_fdw', 'arrow_bytea_le'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION arrow_bytea_ge(arrow_bytea, arrow_bytea) RETURNS bool
AS 'parquet_fdw', 'arrow_bytea_ge'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION arrow_bytea_eq(arrow_bytea, arrow_bytea) RETURNS bool
AS 'parquet_fdw', 'arrow_bytea_eq'
LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR < (
	LEFTARG = arrow_bytea, RIGHTARG = arrow_bytea, PROCEDURE = arrow_bytea_lt,
	COMMUTATOR = '>', NEGATOR = '>='
);

CREATE OPERATOR > (
	LEFTARG = arrow_bytea, RIGHTARG = arrow_bytea, PROCEDURE = arrow_bytea_gt,
	COMMUTATOR = '<', NEGATOR = '<='
);

CREATE OPERATOR <= (
	LEFTARG = arrow_bytea, RIGHTARG = arrow_bytea, PROCEDURE = arrow_bytea_le,
	COMMUTATOR = '>=', NEGATOR = '>'
);

CREATE OPERATOR >= (
	LEFTARG = arrow_bytea, RIGHTARG = arrow_bytea, PROCEDURE = arrow_bytea_ge,
	COMMUTATOR = '<=', NEGATOR = '<'
);

CREATE OPERATOR = (
	LEFTARG = arrow_bytea, RIGHTARG = arrow_bytea, PROCEDURE = arrow_bytea_eq,
	COMMUTATOR = '=', NEGATOR = '<>'
);


CREATE OPERATOR CLASS arrow_bytea_ops
    DEFAULT FOR TYPE arrow_bytea USING btree AS
        OPERATOR        1       < ,
        OPERATOR        2       <= ,
        OPERATOR        3       = ,
        OPERATOR        4       >= ,
        OPERATOR        5       > ,
        FUNCTION        1       arrow_bytea_cmp(arrow_bytea, arrow_bytea);

