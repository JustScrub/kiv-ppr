#ifndef __COMP_CONF_HPP__
#define __COMP_CONF_HPP__

/* ------------ CONFIG ---------------- */
/*             EDIT THIS PART           */
constexpr size_t CHUNK_SIZE = 32768-192; // set to a little less than cache size in bytes (about 192 bytes less)



/* ------------- AUXILARIES ----------- */
/*              DO NOT CHANGE           */
constexpr size_t CHUNK_CAPACITY = CHUNK_SIZE / sizeof(float); 
constexpr size_t VEC_REG_CAP = 8; // 256 bits / 32 bits = 8 floats

#endif // __COMP_CONF_HPP__