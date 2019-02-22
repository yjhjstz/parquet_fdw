#ifndef ARROW_BYTEA_H
#define ARROW_BYTEA_H

typedef struct 
{
    int64_t     len;
    const char *data;
} arrow_bytea;

arrow_bytea *make_arrow_bytea(const char *data, int64_t len);

#endif
