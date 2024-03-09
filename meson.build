project('rk3588-npu', 'c')
incdir = include_directories('include')
lib_src = ['src/npu_interface.c','src/npu_matmul.c']
lib = library('rk3588-npu',lib_src, include_directories : incdir)

test_src = 'tests/matmul_4_36_16.c'
test_matmul  = executable('matmul_4_36_16', test_src, include_directories : incdir, link_with : lib)
test('matmul 4x36x16',test_matmul)
