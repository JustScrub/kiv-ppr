#ifndef __COMP_CONF_HPP__
#define __COMP_CONF_HPP__

/* ------------ CONFIG ---------------- */
/*             EDIT THIS PART           */
constexpr size_t CHUNK_SIZE = 32768; // set to cache size



/* ------------- AUXILARIES ----------- */
/*              DO NOT CHANGE           */
constexpr size_t CHUNK_CAPACITY = CHUNK_SIZE / sizeof(float); // do not change

#endif // __COMP_CONF_HPP__