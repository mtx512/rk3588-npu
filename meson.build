project('rk3588-npu', 'c')
incdir = include_directories('include')
lib_src = ['src/npu_interface.c','src/npu_matmul.c']
lib = library('rk3588-npu',lib_src, include_directories : incdir)


test_matmul_4_36_16  = executable('matmul_4_36_16', 'tests/matmul_4_36_16.c', include_directories : incdir, link_with : lib)
test('matmul_4x36x16',test_matmul_4_36_16, is_parallel : false)

test_matmul_fp16  = executable('matmul_fp16', 'tests/matmul_fp16.c', include_directories : incdir, link_with : lib, link_args : '-lm')
test('matmul fp16 1x32x16',test_matmul_fp16, is_parallel : false , args : ['1', '32' ,'16'])
test('matmul fp16 1x64x64',test_matmul_fp16, is_parallel : false , args : ['1', '64' ,'64'])
test('matmul fp16 1x1024x1024',test_matmul_fp16, is_parallel : false , args : ['1', '1024' ,'1024'])
test('matmul fp16 1x4096x4096',test_matmul_fp16, is_parallel : false , args : ['1', '4096' ,'4096'])
test('matmul fp16 4x32x16',test_matmul_fp16, is_parallel : false , args : ['4', '32' ,'16'])
# test max feature data for one task
test('matmul fp16 384x384x4096',test_matmul_fp16, is_parallel : false , args : ['384', '384' ,'4096'])

test_matmul_int8  = executable('matmul_int8', 'tests/matmul_int8.c', include_directories : incdir, link_with : lib, link_args : '-lm')
test('matmul int8 1x32x32',test_matmul_int8, is_parallel : false , args : ['1','32','32'])
test('matmul int8 1x64x64',test_matmul_int8, is_parallel : false , args : ['1','64','64'])
test('matmul int8 1x1024x1024',test_matmul_int8, is_parallel : false , args : ['1','1024','1024'])
test('matmul int8 1x4096x4096',test_matmul_int8, is_parallel : false , args : ['1','4096','4096'])
# test max feature data for one task
test('matmul int8 544x544x4096',test_matmul_int8, is_parallel : false , args : ['544','544','4096'])
