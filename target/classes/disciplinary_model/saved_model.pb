ď	
§5ţ4
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
.
Log1p
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
î
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
Ľ

ScatterAdd
ref"T
indices"Tindices
updates"T

output_ref"T" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
Á
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.12.02v1.12.0-0-ga6d8ffae09Í	
l
input_xPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
l
input_yPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
shape:˙˙˙˙˙˙˙˙˙¤
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
L
PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙

*embedding/Initializer/random_uniform/shapeConst*
_class
loc:@embedding*
valueB" N  ,  *
dtype0*
_output_shapes
:

(embedding/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
valueB
 *VÖź

(embedding/Initializer/random_uniform/maxConst*
_class
loc:@embedding*
valueB
 *VÖ<*
dtype0*
_output_shapes
: 
ß
2embedding/Initializer/random_uniform/RandomUniformRandomUniform*embedding/Initializer/random_uniform/shape*
seed2 *
dtype0*!
_output_shapes
: Ź*

seed *
T0*
_class
loc:@embedding
Â
(embedding/Initializer/random_uniform/subSub(embedding/Initializer/random_uniform/max(embedding/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@embedding
×
(embedding/Initializer/random_uniform/mulMul2embedding/Initializer/random_uniform/RandomUniform(embedding/Initializer/random_uniform/sub*
_class
loc:@embedding*!
_output_shapes
: Ź*
T0
É
$embedding/Initializer/random_uniformAdd(embedding/Initializer/random_uniform/mul(embedding/Initializer/random_uniform/min*!
_output_shapes
: Ź*
T0*
_class
loc:@embedding
°
	embedding
VariableV2"/device:CPU:0*
dtype0*!
_output_shapes
: Ź*
shared_name *
_class
loc:@embedding*
	container *
shape: Ź
Í
embedding/AssignAssign	embedding$embedding/Initializer/random_uniform"/device:CPU:0*
validate_shape(*!
_output_shapes
: Ź*
use_locking(*
T0*
_class
loc:@embedding
~
embedding/readIdentity	embedding"/device:CPU:0*
T0*
_class
loc:@embedding*!
_output_shapes
: Ź

embedding_1/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
_class
loc:@embedding*
value	B : 
Ę
embedding_1GatherV2embedding/readinput_xembedding_1/axis"/device:CPU:0*
Tindices0*
Tparams0*
_class
loc:@embedding*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
Taxis0
t
embedding_1/IdentityIdentityembedding_1"/device:CPU:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
h
ExpandDims/dimConst"/device:CPU:0*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 


ExpandDims
ExpandDimsembedding_1/IdentityExpandDims/dim"/device:CPU:0*

Tdim0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź

'cnn0/w/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@cnn0/w*%
valueB"   ,        *
dtype0

%cnn0/w/Initializer/random_uniform/minConst*
_class
loc:@cnn0/w*
valueB
 *łfĚť*
dtype0*
_output_shapes
: 

%cnn0/w/Initializer/random_uniform/maxConst*
_class
loc:@cnn0/w*
valueB
 *łfĚ;*
dtype0*
_output_shapes
: 
Ý
/cnn0/w/Initializer/random_uniform/RandomUniformRandomUniform'cnn0/w/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:Ź*

seed *
T0*
_class
loc:@cnn0/w*
seed2 
ś
%cnn0/w/Initializer/random_uniform/subSub%cnn0/w/Initializer/random_uniform/max%cnn0/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn0/w*
_output_shapes
: 
Ň
%cnn0/w/Initializer/random_uniform/mulMul/cnn0/w/Initializer/random_uniform/RandomUniform%cnn0/w/Initializer/random_uniform/sub*
T0*
_class
loc:@cnn0/w*(
_output_shapes
:Ź
Ä
!cnn0/w/Initializer/random_uniformAdd%cnn0/w/Initializer/random_uniform/mul%cnn0/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn0/w*(
_output_shapes
:Ź
Š
cnn0/w
VariableV2*
shape:Ź*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn0/w*
	container 
š
cnn0/w/AssignAssigncnn0/w!cnn0/w/Initializer/random_uniform*
T0*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(
m
cnn0/w/readIdentitycnn0/w*
T0*
_class
loc:@cnn0/w*(
_output_shapes
:Ź
Y

cnn0/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
t
cnn0/b
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

cnn0/b/AssignAssigncnn0/b
cnn0/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(
`
cnn0/b/readIdentitycnn0/b*
T0*
_class
loc:@cnn0/b*
_output_shapes	
:
Ů
cnn0/Conv2DConv2D
ExpandDimscnn0/w/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

cnn0/BiasAddBiasAddcnn0/Conv2Dcnn0/b/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
	cnn0/convRelucnn0/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
cnn0/gmpMaxPool	cnn0/conv*
ksize	
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC*
strides

l
cnn0/SqueezeSqueezecnn0/gmp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims


'cnn1/w/Initializer/random_uniform/shapeConst*
_class
loc:@cnn1/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

%cnn1/w/Initializer/random_uniform/minConst*
_class
loc:@cnn1/w*
valueB
 *äŚť*
dtype0*
_output_shapes
: 

%cnn1/w/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@cnn1/w*
valueB
 *äŚ;
Ý
/cnn1/w/Initializer/random_uniform/RandomUniformRandomUniform'cnn1/w/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@cnn1/w*
seed2 *
dtype0*(
_output_shapes
:Ź
ś
%cnn1/w/Initializer/random_uniform/subSub%cnn1/w/Initializer/random_uniform/max%cnn1/w/Initializer/random_uniform/min*
_class
loc:@cnn1/w*
_output_shapes
: *
T0
Ň
%cnn1/w/Initializer/random_uniform/mulMul/cnn1/w/Initializer/random_uniform/RandomUniform%cnn1/w/Initializer/random_uniform/sub*
T0*
_class
loc:@cnn1/w*(
_output_shapes
:Ź
Ä
!cnn1/w/Initializer/random_uniformAdd%cnn1/w/Initializer/random_uniform/mul%cnn1/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn1/w*(
_output_shapes
:Ź
Š
cnn1/w
VariableV2*
shape:Ź*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn1/w*
	container 
š
cnn1/w/AssignAssigncnn1/w!cnn1/w/Initializer/random_uniform*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn1/w*
validate_shape(
m
cnn1/w/readIdentitycnn1/w*
T0*
_class
loc:@cnn1/w*(
_output_shapes
:Ź
Y

cnn1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
t
cnn1/b
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

cnn1/b/AssignAssigncnn1/b
cnn1/Const*
T0*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:*
use_locking(
`
cnn1/b/readIdentitycnn1/b*
T0*
_class
loc:@cnn1/b*
_output_shapes	
:
Ů
cnn1/Conv2DConv2D
ExpandDimscnn1/w/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

cnn1/BiasAddBiasAddcnn1/Conv2Dcnn1/b/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
	cnn1/convRelucnn1/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
cnn1/gmpMaxPool	cnn1/conv*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
cnn1/SqueezeSqueezecnn1/gmp*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0

'cnn2/w/Initializer/random_uniform/shapeConst*
_class
loc:@cnn2/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

%cnn2/w/Initializer/random_uniform/minConst*
_class
loc:@cnn2/w*
valueB
 *ť*
dtype0*
_output_shapes
: 

%cnn2/w/Initializer/random_uniform/maxConst*
_class
loc:@cnn2/w*
valueB
 *;*
dtype0*
_output_shapes
: 
Ý
/cnn2/w/Initializer/random_uniform/RandomUniformRandomUniform'cnn2/w/Initializer/random_uniform/shape*
seed2 *
dtype0*(
_output_shapes
:Ź*

seed *
T0*
_class
loc:@cnn2/w
ś
%cnn2/w/Initializer/random_uniform/subSub%cnn2/w/Initializer/random_uniform/max%cnn2/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn2/w*
_output_shapes
: 
Ň
%cnn2/w/Initializer/random_uniform/mulMul/cnn2/w/Initializer/random_uniform/RandomUniform%cnn2/w/Initializer/random_uniform/sub*
_class
loc:@cnn2/w*(
_output_shapes
:Ź*
T0
Ä
!cnn2/w/Initializer/random_uniformAdd%cnn2/w/Initializer/random_uniform/mul%cnn2/w/Initializer/random_uniform/min*(
_output_shapes
:Ź*
T0*
_class
loc:@cnn2/w
Š
cnn2/w
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn2/w*
	container *
shape:Ź
š
cnn2/w/AssignAssigncnn2/w!cnn2/w/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź
m
cnn2/w/readIdentitycnn2/w*
T0*
_class
loc:@cnn2/w*(
_output_shapes
:Ź
Y

cnn2/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
t
cnn2/b
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

cnn2/b/AssignAssigncnn2/b
cnn2/Const*
T0*
_class
loc:@cnn2/b*
validate_shape(*
_output_shapes	
:*
use_locking(
`
cnn2/b/readIdentitycnn2/b*
T0*
_class
loc:@cnn2/b*
_output_shapes	
:
Ů
cnn2/Conv2DConv2D
ExpandDimscnn2/w/read*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

cnn2/BiasAddBiasAddcnn2/Conv2Dcnn2/b/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
	cnn2/convRelucnn2/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
cnn2/gmpMaxPool	cnn2/conv*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
cnn2/SqueezeSqueezecnn2/gmp*
squeeze_dims
*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

'cnn3/w/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@cnn3/w*%
valueB"   ,        

%cnn3/w/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@cnn3/w*
valueB
 *OFť*
dtype0

%cnn3/w/Initializer/random_uniform/maxConst*
_class
loc:@cnn3/w*
valueB
 *OF;*
dtype0*
_output_shapes
: 
Ý
/cnn3/w/Initializer/random_uniform/RandomUniformRandomUniform'cnn3/w/Initializer/random_uniform/shape*
seed2 *
dtype0*(
_output_shapes
:Ź*

seed *
T0*
_class
loc:@cnn3/w
ś
%cnn3/w/Initializer/random_uniform/subSub%cnn3/w/Initializer/random_uniform/max%cnn3/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn3/w*
_output_shapes
: 
Ň
%cnn3/w/Initializer/random_uniform/mulMul/cnn3/w/Initializer/random_uniform/RandomUniform%cnn3/w/Initializer/random_uniform/sub*(
_output_shapes
:Ź*
T0*
_class
loc:@cnn3/w
Ä
!cnn3/w/Initializer/random_uniformAdd%cnn3/w/Initializer/random_uniform/mul%cnn3/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn3/w*(
_output_shapes
:Ź
Š
cnn3/w
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn3/w*
	container *
shape:Ź
š
cnn3/w/AssignAssigncnn3/w!cnn3/w/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź
m
cnn3/w/readIdentitycnn3/w*
T0*
_class
loc:@cnn3/w*(
_output_shapes
:Ź
Y

cnn3/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
t
cnn3/b
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

cnn3/b/AssignAssigncnn3/b
cnn3/Const*
use_locking(*
T0*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:
`
cnn3/b/readIdentitycnn3/b*
T0*
_class
loc:@cnn3/b*
_output_shapes	
:
Ů
cnn3/Conv2DConv2D
ExpandDimscnn3/w/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0

cnn3/BiasAddBiasAddcnn3/Conv2Dcnn3/b/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
	cnn3/convRelucnn3/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
cnn3/gmpMaxPool	cnn3/conv*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID
l
cnn3/SqueezeSqueezecnn3/gmp*
squeeze_dims
*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

'cnn4/w/Initializer/random_uniform/shapeConst*
_class
loc:@cnn4/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

%cnn4/w/Initializer/random_uniform/minConst*
_class
loc:@cnn4/w*
valueB
 *Ťlť*
dtype0*
_output_shapes
: 

%cnn4/w/Initializer/random_uniform/maxConst*
_class
loc:@cnn4/w*
valueB
 *Ťl;*
dtype0*
_output_shapes
: 
Ý
/cnn4/w/Initializer/random_uniform/RandomUniformRandomUniform'cnn4/w/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:Ź*

seed *
T0*
_class
loc:@cnn4/w*
seed2 
ś
%cnn4/w/Initializer/random_uniform/subSub%cnn4/w/Initializer/random_uniform/max%cnn4/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn4/w*
_output_shapes
: 
Ň
%cnn4/w/Initializer/random_uniform/mulMul/cnn4/w/Initializer/random_uniform/RandomUniform%cnn4/w/Initializer/random_uniform/sub*
T0*
_class
loc:@cnn4/w*(
_output_shapes
:Ź
Ä
!cnn4/w/Initializer/random_uniformAdd%cnn4/w/Initializer/random_uniform/mul%cnn4/w/Initializer/random_uniform/min*
T0*
_class
loc:@cnn4/w*(
_output_shapes
:Ź
Š
cnn4/w
VariableV2*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn4/w*
	container *
shape:Ź*
dtype0
š
cnn4/w/AssignAssigncnn4/w!cnn4/w/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź
m
cnn4/w/readIdentitycnn4/w*(
_output_shapes
:Ź*
T0*
_class
loc:@cnn4/w
Y

cnn4/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
t
cnn4/b
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 

cnn4/b/AssignAssigncnn4/b
cnn4/Const*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(
`
cnn4/b/readIdentitycnn4/b*
T0*
_class
loc:@cnn4/b*
_output_shapes	
:
Ů
cnn4/Conv2DConv2D
ExpandDimscnn4/w/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

cnn4/BiasAddBiasAddcnn4/Conv2Dcnn4/b/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
	cnn4/convRelucnn4/BiasAdd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
cnn4/gmpMaxPool	cnn4/conv*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
cnn4/SqueezeSqueezecnn4/gmp*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
V
concat/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
­
concatConcatV2cnn0/Squeezecnn1/Squeezecnn2/Squeezecnn3/Squeezecnn4/Squeezeconcat/axis*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

Tidx0*
T0*
N
^
Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
j
ReshapeReshapeconcatReshape/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
Tshape0

-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"      *
dtype0

+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *m˝*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *m=*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:

*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:


Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:


Ľ
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:

*
dtype0* 
_output_shapes
:


É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(* 
_output_shapes
:


w
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:



dense/bias/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@dense/bias*
valueB*    *
dtype0


dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@dense/bias*
	container *
shape:
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:
l
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes	
:

score/dense/MatMulMatMulReshapedense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

score/dense/BiasAddBiasAddscore/dense/MatMuldense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
score/dense/ReluReluscore/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

"w/Initializer/random_uniform/shapeConst*
_class

loc:@w*
valueB"   ¤  *
dtype0*
_output_shapes
:
{
 w/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class

loc:@w*
valueB
 *R¤˝
{
 w/Initializer/random_uniform/maxConst*
_class

loc:@w*
valueB
 *R¤=*
dtype0*
_output_shapes
: 
Ć
*w/Initializer/random_uniform/RandomUniformRandomUniform"w/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
¤*

seed *
T0*
_class

loc:@w*
seed2 
˘
 w/Initializer/random_uniform/subSub w/Initializer/random_uniform/max w/Initializer/random_uniform/min*
T0*
_class

loc:@w*
_output_shapes
: 
ś
 w/Initializer/random_uniform/mulMul*w/Initializer/random_uniform/RandomUniform w/Initializer/random_uniform/sub* 
_output_shapes
:
¤*
T0*
_class

loc:@w
¨
w/Initializer/random_uniformAdd w/Initializer/random_uniform/mul w/Initializer/random_uniform/min*
T0*
_class

loc:@w* 
_output_shapes
:
¤

w
VariableV2*
dtype0* 
_output_shapes
:
¤*
shared_name *
_class

loc:@w*
	container *
shape:
¤

w/AssignAssignww/Initializer/random_uniform* 
_output_shapes
:
¤*
use_locking(*
T0*
_class

loc:@w*
validate_shape(
V
w/readIdentityw*
T0*
_class

loc:@w* 
_output_shapes
:
¤
Z
score/ConstConst*
valueB¤*    *
dtype0*
_output_shapes	
:¤
u
score/b
VariableV2*
dtype0*
_output_shapes	
:¤*
	container *
shape:¤*
shared_name 

score/b/AssignAssignscore/bscore/Const*
validate_shape(*
_output_shapes	
:¤*
use_locking(*
T0*
_class
loc:@score/b
c
score/b/readIdentityscore/b*
T0*
_class
loc:@score/b*
_output_shapes	
:¤

score/MatMulMatMulscore/dense/Reluw/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
transpose_a( *
transpose_b( 
~
score/BiasAddBiasAddscore/MatMulscore/b/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
`
score/dropout/ShapeShapescore/BiasAdd*
T0*
out_type0*
_output_shapes
:
e
 score/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
e
 score/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Š
*score/dropout/random_uniform/RandomUniformRandomUniformscore/dropout/Shape*

seed *
T0*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
seed2 

 score/dropout/random_uniform/subSub score/dropout/random_uniform/max score/dropout/random_uniform/min*
T0*
_output_shapes
: 
¨
 score/dropout/random_uniform/mulMul*score/dropout/random_uniform/RandomUniform score/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0

score/dropout/random_uniformAdd score/dropout/random_uniform/mul score/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
d
score/dropout/addAdd	keep_probscore/dropout/random_uniform*
_output_shapes
:*
T0
R
score/dropout/FloorFloorscore/dropout/add*
_output_shapes
:*
T0
Y
score/dropout/divRealDivscore/BiasAdd	keep_prob*
T0*
_output_shapes
:
s
score/dropout/mulMulscore/dropout/divscore/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
^
score/SigmoidSigmoidscore/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
t
!optimize/logistic_loss/zeros_like	ZerosLikescore/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0

#optimize/logistic_loss/GreaterEqualGreaterEqualscore/dropout/mul!optimize/logistic_loss/zeros_like*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
ľ
optimize/logistic_loss/SelectSelect#optimize/logistic_loss/GreaterEqualscore/dropout/mul!optimize/logistic_loss/zeros_like*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
g
optimize/logistic_loss/NegNegscore/dropout/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
°
optimize/logistic_loss/Select_1Select#optimize/logistic_loss/GreaterEqualoptimize/logistic_loss/Negscore/dropout/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
p
optimize/logistic_loss/mulMulscore/dropout/mulinput_y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

optimize/logistic_loss/subSuboptimize/logistic_loss/Selectoptimize/logistic_loss/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
u
optimize/logistic_loss/ExpExpoptimize/logistic_loss/Select_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
t
optimize/logistic_loss/Log1pLog1poptimize/logistic_loss/Exp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

optimize/logistic_lossAddoptimize/logistic_loss/suboptimize/logistic_loss/Log1p*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
_
optimize/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
{
optimize/MeanMeanoptimize/logistic_lossoptimize/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
optimize/Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
u
optimize/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ć
optimize/Variable/AssignAssignoptimize/Variableoptimize/Variable/initial_value*$
_class
loc:@optimize/Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
|
optimize/Variable/readIdentityoptimize/Variable*
T0*$
_class
loc:@optimize/Variable*
_output_shapes
: 
l
'optimize/ExponentialDecay/learning_rateConst*
_output_shapes
: *
valueB
 *ˇQ8*
dtype0
d
 optimize/ExponentialDecay/Cast/xConst*
valueB	 : *
dtype0*
_output_shapes
: 

optimize/ExponentialDecay/CastCast optimize/ExponentialDecay/Cast/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
d
"optimize/ExponentialDecay/Cast_1/xConst*
value	B :*
dtype0*
_output_shapes
: 

 optimize/ExponentialDecay/Cast_1Cast"optimize/ExponentialDecay/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 

 optimize/ExponentialDecay/Cast_2Castoptimize/Variable/read*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

!optimize/ExponentialDecay/truedivRealDiv optimize/ExponentialDecay/Cast_2optimize/ExponentialDecay/Cast*
T0*
_output_shapes
: 

optimize/ExponentialDecay/PowPow optimize/ExponentialDecay/Cast_1!optimize/ExponentialDecay/truediv*
T0*
_output_shapes
: 

optimize/ExponentialDecayMul'optimize/ExponentialDecay/learning_rateoptimize/ExponentialDecay/Pow*
T0*
_output_shapes
: 
[
optimize/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
optimize/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

optimize/gradients/FillFilloptimize/gradients/Shapeoptimize/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0

3optimize/gradients/optimize/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
˝
-optimize/gradients/optimize/Mean_grad/ReshapeReshapeoptimize/gradients/Fill3optimize/gradients/optimize/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0

+optimize/gradients/optimize/Mean_grad/ShapeShapeoptimize/logistic_loss*
T0*
out_type0*
_output_shapes
:
Ó
*optimize/gradients/optimize/Mean_grad/TileTile-optimize/gradients/optimize/Mean_grad/Reshape+optimize/gradients/optimize/Mean_grad/Shape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*

Tmultiples0

-optimize/gradients/optimize/Mean_grad/Shape_1Shapeoptimize/logistic_loss*
_output_shapes
:*
T0*
out_type0
p
-optimize/gradients/optimize/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
u
+optimize/gradients/optimize/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ě
*optimize/gradients/optimize/Mean_grad/ProdProd-optimize/gradients/optimize/Mean_grad/Shape_1+optimize/gradients/optimize/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
w
-optimize/gradients/optimize/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Đ
,optimize/gradients/optimize/Mean_grad/Prod_1Prod-optimize/gradients/optimize/Mean_grad/Shape_2-optimize/gradients/optimize/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
q
/optimize/gradients/optimize/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
¸
-optimize/gradients/optimize/Mean_grad/MaximumMaximum,optimize/gradients/optimize/Mean_grad/Prod_1/optimize/gradients/optimize/Mean_grad/Maximum/y*
_output_shapes
: *
T0
ś
.optimize/gradients/optimize/Mean_grad/floordivFloorDiv*optimize/gradients/optimize/Mean_grad/Prod-optimize/gradients/optimize/Mean_grad/Maximum*
T0*
_output_shapes
: 
˘
*optimize/gradients/optimize/Mean_grad/CastCast.optimize/gradients/optimize/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Ă
-optimize/gradients/optimize/Mean_grad/truedivRealDiv*optimize/gradients/optimize/Mean_grad/Tile*optimize/gradients/optimize/Mean_grad/Cast*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

4optimize/gradients/optimize/logistic_loss_grad/ShapeShapeoptimize/logistic_loss/sub*
T0*
out_type0*
_output_shapes
:

6optimize/gradients/optimize/logistic_loss_grad/Shape_1Shapeoptimize/logistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:

Doptimize/gradients/optimize/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs4optimize/gradients/optimize/logistic_loss_grad/Shape6optimize/gradients/optimize/logistic_loss_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
î
2optimize/gradients/optimize/logistic_loss_grad/SumSum-optimize/gradients/optimize/Mean_grad/truedivDoptimize/gradients/optimize/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ě
6optimize/gradients/optimize/logistic_loss_grad/ReshapeReshape2optimize/gradients/optimize/logistic_loss_grad/Sum4optimize/gradients/optimize/logistic_loss_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
ň
4optimize/gradients/optimize/logistic_loss_grad/Sum_1Sum-optimize/gradients/optimize/Mean_grad/truedivFoptimize/gradients/optimize/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ň
8optimize/gradients/optimize/logistic_loss_grad/Reshape_1Reshape4optimize/gradients/optimize/logistic_loss_grad/Sum_16optimize/gradients/optimize/logistic_loss_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
ť
?optimize/gradients/optimize/logistic_loss_grad/tuple/group_depsNoOp7^optimize/gradients/optimize/logistic_loss_grad/Reshape9^optimize/gradients/optimize/logistic_loss_grad/Reshape_1
Ë
Goptimize/gradients/optimize/logistic_loss_grad/tuple/control_dependencyIdentity6optimize/gradients/optimize/logistic_loss_grad/Reshape@^optimize/gradients/optimize/logistic_loss_grad/tuple/group_deps*
T0*I
_class?
=;loc:@optimize/gradients/optimize/logistic_loss_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
Ń
Ioptimize/gradients/optimize/logistic_loss_grad/tuple/control_dependency_1Identity8optimize/gradients/optimize/logistic_loss_grad/Reshape_1@^optimize/gradients/optimize/logistic_loss_grad/tuple/group_deps*
T0*K
_classA
?=loc:@optimize/gradients/optimize/logistic_loss_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

8optimize/gradients/optimize/logistic_loss/sub_grad/ShapeShapeoptimize/logistic_loss/Select*
T0*
out_type0*
_output_shapes
:

:optimize/gradients/optimize/logistic_loss/sub_grad/Shape_1Shapeoptimize/logistic_loss/mul*
T0*
out_type0*
_output_shapes
:

Hoptimize/gradients/optimize/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8optimize/gradients/optimize/logistic_loss/sub_grad/Shape:optimize/gradients/optimize/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

6optimize/gradients/optimize/logistic_loss/sub_grad/SumSumGoptimize/gradients/optimize/logistic_loss_grad/tuple/control_dependencyHoptimize/gradients/optimize/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ř
:optimize/gradients/optimize/logistic_loss/sub_grad/ReshapeReshape6optimize/gradients/optimize/logistic_loss/sub_grad/Sum8optimize/gradients/optimize/logistic_loss/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

8optimize/gradients/optimize/logistic_loss/sub_grad/Sum_1SumGoptimize/gradients/optimize/logistic_loss_grad/tuple/control_dependencyJoptimize/gradients/optimize/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

6optimize/gradients/optimize/logistic_loss/sub_grad/NegNeg8optimize/gradients/optimize/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
ü
<optimize/gradients/optimize/logistic_loss/sub_grad/Reshape_1Reshape6optimize/gradients/optimize/logistic_loss/sub_grad/Neg:optimize/gradients/optimize/logistic_loss/sub_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0*
Tshape0
Ç
Coptimize/gradients/optimize/logistic_loss/sub_grad/tuple/group_depsNoOp;^optimize/gradients/optimize/logistic_loss/sub_grad/Reshape=^optimize/gradients/optimize/logistic_loss/sub_grad/Reshape_1
Ű
Koptimize/gradients/optimize/logistic_loss/sub_grad/tuple/control_dependencyIdentity:optimize/gradients/optimize/logistic_loss/sub_grad/ReshapeD^optimize/gradients/optimize/logistic_loss/sub_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0*M
_classC
A?loc:@optimize/gradients/optimize/logistic_loss/sub_grad/Reshape
á
Moptimize/gradients/optimize/logistic_loss/sub_grad/tuple/control_dependency_1Identity<optimize/gradients/optimize/logistic_loss/sub_grad/Reshape_1D^optimize/gradients/optimize/logistic_loss/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimize/gradients/optimize/logistic_loss/sub_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
Ë
:optimize/gradients/optimize/logistic_loss/Log1p_grad/add/xConstJ^optimize/gradients/optimize/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ę
8optimize/gradients/optimize/logistic_loss/Log1p_grad/addAdd:optimize/gradients/optimize/logistic_loss/Log1p_grad/add/xoptimize/logistic_loss/Exp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
ş
?optimize/gradients/optimize/logistic_loss/Log1p_grad/Reciprocal
Reciprocal8optimize/gradients/optimize/logistic_loss/Log1p_grad/add*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
ţ
8optimize/gradients/optimize/logistic_loss/Log1p_grad/mulMulIoptimize/gradients/optimize/logistic_loss_grad/tuple/control_dependency_1?optimize/gradients/optimize/logistic_loss/Log1p_grad/Reciprocal*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

@optimize/gradients/optimize/logistic_loss/Select_grad/zeros_like	ZerosLikescore/dropout/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
­
<optimize/gradients/optimize/logistic_loss/Select_grad/SelectSelect#optimize/logistic_loss/GreaterEqualKoptimize/gradients/optimize/logistic_loss/sub_grad/tuple/control_dependency@optimize/gradients/optimize/logistic_loss/Select_grad/zeros_like*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
Ż
>optimize/gradients/optimize/logistic_loss/Select_grad/Select_1Select#optimize/logistic_loss/GreaterEqual@optimize/gradients/optimize/logistic_loss/Select_grad/zeros_likeKoptimize/gradients/optimize/logistic_loss/sub_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
Î
Foptimize/gradients/optimize/logistic_loss/Select_grad/tuple/group_depsNoOp=^optimize/gradients/optimize/logistic_loss/Select_grad/Select?^optimize/gradients/optimize/logistic_loss/Select_grad/Select_1
ĺ
Noptimize/gradients/optimize/logistic_loss/Select_grad/tuple/control_dependencyIdentity<optimize/gradients/optimize/logistic_loss/Select_grad/SelectG^optimize/gradients/optimize/logistic_loss/Select_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimize/gradients/optimize/logistic_loss/Select_grad/Select*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
ë
Poptimize/gradients/optimize/logistic_loss/Select_grad/tuple/control_dependency_1Identity>optimize/gradients/optimize/logistic_loss/Select_grad/Select_1G^optimize/gradients/optimize/logistic_loss/Select_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@optimize/gradients/optimize/logistic_loss/Select_grad/Select_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

8optimize/gradients/optimize/logistic_loss/mul_grad/ShapeShapescore/dropout/mul*
out_type0*
_output_shapes
:*
T0

:optimize/gradients/optimize/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0*
_output_shapes
:

Hoptimize/gradients/optimize/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8optimize/gradients/optimize/logistic_loss/mul_grad/Shape:optimize/gradients/optimize/logistic_loss/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Č
6optimize/gradients/optimize/logistic_loss/mul_grad/MulMulMoptimize/gradients/optimize/logistic_loss/sub_grad/tuple/control_dependency_1input_y*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
˙
6optimize/gradients/optimize/logistic_loss/mul_grad/SumSum6optimize/gradients/optimize/logistic_loss/mul_grad/MulHoptimize/gradients/optimize/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ř
:optimize/gradients/optimize/logistic_loss/mul_grad/ReshapeReshape6optimize/gradients/optimize/logistic_loss/mul_grad/Sum8optimize/gradients/optimize/logistic_loss/mul_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
Ô
8optimize/gradients/optimize/logistic_loss/mul_grad/Mul_1Mulscore/dropout/mulMoptimize/gradients/optimize/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

8optimize/gradients/optimize/logistic_loss/mul_grad/Sum_1Sum8optimize/gradients/optimize/logistic_loss/mul_grad/Mul_1Joptimize/gradients/optimize/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ţ
<optimize/gradients/optimize/logistic_loss/mul_grad/Reshape_1Reshape8optimize/gradients/optimize/logistic_loss/mul_grad/Sum_1:optimize/gradients/optimize/logistic_loss/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0*
Tshape0
Ç
Coptimize/gradients/optimize/logistic_loss/mul_grad/tuple/group_depsNoOp;^optimize/gradients/optimize/logistic_loss/mul_grad/Reshape=^optimize/gradients/optimize/logistic_loss/mul_grad/Reshape_1
Ű
Koptimize/gradients/optimize/logistic_loss/mul_grad/tuple/control_dependencyIdentity:optimize/gradients/optimize/logistic_loss/mul_grad/ReshapeD^optimize/gradients/optimize/logistic_loss/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@optimize/gradients/optimize/logistic_loss/mul_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
á
Moptimize/gradients/optimize/logistic_loss/mul_grad/tuple/control_dependency_1Identity<optimize/gradients/optimize/logistic_loss/mul_grad/Reshape_1D^optimize/gradients/optimize/logistic_loss/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimize/gradients/optimize/logistic_loss/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
Ć
6optimize/gradients/optimize/logistic_loss/Exp_grad/mulMul8optimize/gradients/optimize/logistic_loss/Log1p_grad/muloptimize/logistic_loss/Exp*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0

Boptimize/gradients/optimize/logistic_loss/Select_1_grad/zeros_like	ZerosLikeoptimize/logistic_loss/Neg*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

>optimize/gradients/optimize/logistic_loss/Select_1_grad/SelectSelect#optimize/logistic_loss/GreaterEqual6optimize/gradients/optimize/logistic_loss/Exp_grad/mulBoptimize/gradients/optimize/logistic_loss/Select_1_grad/zeros_like*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

@optimize/gradients/optimize/logistic_loss/Select_1_grad/Select_1Select#optimize/logistic_loss/GreaterEqualBoptimize/gradients/optimize/logistic_loss/Select_1_grad/zeros_like6optimize/gradients/optimize/logistic_loss/Exp_grad/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
Ô
Hoptimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/group_depsNoOp?^optimize/gradients/optimize/logistic_loss/Select_1_grad/SelectA^optimize/gradients/optimize/logistic_loss/Select_1_grad/Select_1
í
Poptimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity>optimize/gradients/optimize/logistic_loss/Select_1_grad/SelectI^optimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@optimize/gradients/optimize/logistic_loss/Select_1_grad/Select*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
ó
Roptimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity@optimize/gradients/optimize/logistic_loss/Select_1_grad/Select_1I^optimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@optimize/gradients/optimize/logistic_loss/Select_1_grad/Select_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
Â
6optimize/gradients/optimize/logistic_loss/Neg_grad/NegNegPoptimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤*
T0
Ő
optimize/gradients/AddNAddNNoptimize/gradients/optimize/logistic_loss/Select_grad/tuple/control_dependencyKoptimize/gradients/optimize/logistic_loss/mul_grad/tuple/control_dependencyRoptimize/gradients/optimize/logistic_loss/Select_1_grad/tuple/control_dependency_16optimize/gradients/optimize/logistic_loss/Neg_grad/Neg*
T0*O
_classE
CAloc:@optimize/gradients/optimize/logistic_loss/Select_grad/Select*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

/optimize/gradients/score/dropout/mul_grad/ShapeShapescore/dropout/div*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
out_type0

1optimize/gradients/score/dropout/mul_grad/Shape_1Shapescore/dropout/Floor*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
?optimize/gradients/score/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs/optimize/gradients/score/dropout/mul_grad/Shape1optimize/gradients/score/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

-optimize/gradients/score/dropout/mul_grad/MulMuloptimize/gradients/AddNscore/dropout/Floor*
_output_shapes
:*
T0
ä
-optimize/gradients/score/dropout/mul_grad/SumSum-optimize/gradients/score/dropout/mul_grad/Mul?optimize/gradients/score/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Í
1optimize/gradients/score/dropout/mul_grad/ReshapeReshape-optimize/gradients/score/dropout/mul_grad/Sum/optimize/gradients/score/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

/optimize/gradients/score/dropout/mul_grad/Mul_1Mulscore/dropout/divoptimize/gradients/AddN*
T0*
_output_shapes
:
ę
/optimize/gradients/score/dropout/mul_grad/Sum_1Sum/optimize/gradients/score/dropout/mul_grad/Mul_1Aoptimize/gradients/score/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ó
3optimize/gradients/score/dropout/mul_grad/Reshape_1Reshape/optimize/gradients/score/dropout/mul_grad/Sum_11optimize/gradients/score/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ź
:optimize/gradients/score/dropout/mul_grad/tuple/group_depsNoOp2^optimize/gradients/score/dropout/mul_grad/Reshape4^optimize/gradients/score/dropout/mul_grad/Reshape_1
§
Boptimize/gradients/score/dropout/mul_grad/tuple/control_dependencyIdentity1optimize/gradients/score/dropout/mul_grad/Reshape;^optimize/gradients/score/dropout/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@optimize/gradients/score/dropout/mul_grad/Reshape*
_output_shapes
:
­
Doptimize/gradients/score/dropout/mul_grad/tuple/control_dependency_1Identity3optimize/gradients/score/dropout/mul_grad/Reshape_1;^optimize/gradients/score/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@optimize/gradients/score/dropout/mul_grad/Reshape_1
|
/optimize/gradients/score/dropout/div_grad/ShapeShapescore/BiasAdd*
T0*
out_type0*
_output_shapes
:

1optimize/gradients/score/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
?optimize/gradients/score/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs/optimize/gradients/score/dropout/div_grad/Shape1optimize/gradients/score/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
1optimize/gradients/score/dropout/div_grad/RealDivRealDivBoptimize/gradients/score/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
č
-optimize/gradients/score/dropout/div_grad/SumSum1optimize/gradients/score/dropout/div_grad/RealDiv?optimize/gradients/score/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ý
1optimize/gradients/score/dropout/div_grad/ReshapeReshape-optimize/gradients/score/dropout/div_grad/Sum/optimize/gradients/score/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
v
-optimize/gradients/score/dropout/div_grad/NegNegscore/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤

3optimize/gradients/score/dropout/div_grad/RealDiv_1RealDiv-optimize/gradients/score/dropout/div_grad/Neg	keep_prob*
_output_shapes
:*
T0
Ą
3optimize/gradients/score/dropout/div_grad/RealDiv_2RealDiv3optimize/gradients/score/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
Đ
-optimize/gradients/score/dropout/div_grad/mulMulBoptimize/gradients/score/dropout/mul_grad/tuple/control_dependency3optimize/gradients/score/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
č
/optimize/gradients/score/dropout/div_grad/Sum_1Sum-optimize/gradients/score/dropout/div_grad/mulAoptimize/gradients/score/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ó
3optimize/gradients/score/dropout/div_grad/Reshape_1Reshape/optimize/gradients/score/dropout/div_grad/Sum_11optimize/gradients/score/dropout/div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
Ź
:optimize/gradients/score/dropout/div_grad/tuple/group_depsNoOp2^optimize/gradients/score/dropout/div_grad/Reshape4^optimize/gradients/score/dropout/div_grad/Reshape_1
ˇ
Boptimize/gradients/score/dropout/div_grad/tuple/control_dependencyIdentity1optimize/gradients/score/dropout/div_grad/Reshape;^optimize/gradients/score/dropout/div_grad/tuple/group_deps*
T0*D
_class:
86loc:@optimize/gradients/score/dropout/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
­
Doptimize/gradients/score/dropout/div_grad/tuple/control_dependency_1Identity3optimize/gradients/score/dropout/div_grad/Reshape_1;^optimize/gradients/score/dropout/div_grad/tuple/group_deps*
T0*F
_class<
:8loc:@optimize/gradients/score/dropout/div_grad/Reshape_1*
_output_shapes
:
Á
1optimize/gradients/score/BiasAdd_grad/BiasAddGradBiasAddGradBoptimize/gradients/score/dropout/div_grad/tuple/control_dependency*
_output_shapes	
:¤*
T0*
data_formatNHWC
ˇ
6optimize/gradients/score/BiasAdd_grad/tuple/group_depsNoOp2^optimize/gradients/score/BiasAdd_grad/BiasAddGradC^optimize/gradients/score/dropout/div_grad/tuple/control_dependency
Ŕ
>optimize/gradients/score/BiasAdd_grad/tuple/control_dependencyIdentityBoptimize/gradients/score/dropout/div_grad/tuple/control_dependency7^optimize/gradients/score/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@optimize/gradients/score/dropout/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
¤
@optimize/gradients/score/BiasAdd_grad/tuple/control_dependency_1Identity1optimize/gradients/score/BiasAdd_grad/BiasAddGrad7^optimize/gradients/score/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@optimize/gradients/score/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:¤
Ö
+optimize/gradients/score/MatMul_grad/MatMulMatMul>optimize/gradients/score/BiasAdd_grad/tuple/control_dependencyw/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ú
-optimize/gradients/score/MatMul_grad/MatMul_1MatMulscore/dense/Relu>optimize/gradients/score/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
¤*
transpose_a(*
transpose_b( *
T0

5optimize/gradients/score/MatMul_grad/tuple/group_depsNoOp,^optimize/gradients/score/MatMul_grad/MatMul.^optimize/gradients/score/MatMul_grad/MatMul_1
Ą
=optimize/gradients/score/MatMul_grad/tuple/control_dependencyIdentity+optimize/gradients/score/MatMul_grad/MatMul6^optimize/gradients/score/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@optimize/gradients/score/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?optimize/gradients/score/MatMul_grad/tuple/control_dependency_1Identity-optimize/gradients/score/MatMul_grad/MatMul_16^optimize/gradients/score/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@optimize/gradients/score/MatMul_grad/MatMul_1* 
_output_shapes
:
¤
Á
1optimize/gradients/score/dense/Relu_grad/ReluGradReluGrad=optimize/gradients/score/MatMul_grad/tuple/control_dependencyscore/dense/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
7optimize/gradients/score/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1optimize/gradients/score/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
˛
<optimize/gradients/score/dense/BiasAdd_grad/tuple/group_depsNoOp8^optimize/gradients/score/dense/BiasAdd_grad/BiasAddGrad2^optimize/gradients/score/dense/Relu_grad/ReluGrad
ť
Doptimize/gradients/score/dense/BiasAdd_grad/tuple/control_dependencyIdentity1optimize/gradients/score/dense/Relu_grad/ReluGrad=^optimize/gradients/score/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@optimize/gradients/score/dense/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
Foptimize/gradients/score/dense/BiasAdd_grad/tuple/control_dependency_1Identity7optimize/gradients/score/dense/BiasAdd_grad/BiasAddGrad=^optimize/gradients/score/dense/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@optimize/gradients/score/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
1optimize/gradients/score/dense/MatMul_grad/MatMulMatMulDoptimize/gradients/score/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( 
Ý
3optimize/gradients/score/dense/MatMul_grad/MatMul_1MatMulReshapeDoptimize/gradients/score/dense/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:

*
transpose_a(*
transpose_b( *
T0
­
;optimize/gradients/score/dense/MatMul_grad/tuple/group_depsNoOp2^optimize/gradients/score/dense/MatMul_grad/MatMul4^optimize/gradients/score/dense/MatMul_grad/MatMul_1
š
Coptimize/gradients/score/dense/MatMul_grad/tuple/control_dependencyIdentity1optimize/gradients/score/dense/MatMul_grad/MatMul<^optimize/gradients/score/dense/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@optimize/gradients/score/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ˇ
Eoptimize/gradients/score/dense/MatMul_grad/tuple/control_dependency_1Identity3optimize/gradients/score/dense/MatMul_grad/MatMul_1<^optimize/gradients/score/dense/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@optimize/gradients/score/dense/MatMul_grad/MatMul_1* 
_output_shapes
:


k
%optimize/gradients/Reshape_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
ß
'optimize/gradients/Reshape_grad/ReshapeReshapeCoptimize/gradients/score/dense/MatMul_grad/tuple/control_dependency%optimize/gradients/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

e
#optimize/gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 

"optimize/gradients/concat_grad/modFloorModconcat/axis#optimize/gradients/concat_grad/Rank*
T0*
_output_shapes
: 
p
$optimize/gradients/concat_grad/ShapeShapecnn0/Squeeze*
_output_shapes
:*
T0*
out_type0
Ë
%optimize/gradients/concat_grad/ShapeNShapeNcnn0/Squeezecnn1/Squeezecnn2/Squeezecnn3/Squeezecnn4/Squeeze*2
_output_shapes 
:::::*
T0*
out_type0*
N
ç
+optimize/gradients/concat_grad/ConcatOffsetConcatOffset"optimize/gradients/concat_grad/mod%optimize/gradients/concat_grad/ShapeN'optimize/gradients/concat_grad/ShapeN:1'optimize/gradients/concat_grad/ShapeN:2'optimize/gradients/concat_grad/ShapeN:3'optimize/gradients/concat_grad/ShapeN:4*
N*2
_output_shapes 
:::::
ę
$optimize/gradients/concat_grad/SliceSlice'optimize/gradients/Reshape_grad/Reshape+optimize/gradients/concat_grad/ConcatOffset%optimize/gradients/concat_grad/ShapeN*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
&optimize/gradients/concat_grad/Slice_1Slice'optimize/gradients/Reshape_grad/Reshape-optimize/gradients/concat_grad/ConcatOffset:1'optimize/gradients/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
&optimize/gradients/concat_grad/Slice_2Slice'optimize/gradients/Reshape_grad/Reshape-optimize/gradients/concat_grad/ConcatOffset:2'optimize/gradients/concat_grad/ShapeN:2*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
&optimize/gradients/concat_grad/Slice_3Slice'optimize/gradients/Reshape_grad/Reshape-optimize/gradients/concat_grad/ConcatOffset:3'optimize/gradients/concat_grad/ShapeN:3*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
&optimize/gradients/concat_grad/Slice_4Slice'optimize/gradients/Reshape_grad/Reshape-optimize/gradients/concat_grad/ConcatOffset:4'optimize/gradients/concat_grad/ShapeN:4*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
T0

/optimize/gradients/concat_grad/tuple/group_depsNoOp%^optimize/gradients/concat_grad/Slice'^optimize/gradients/concat_grad/Slice_1'^optimize/gradients/concat_grad/Slice_2'^optimize/gradients/concat_grad/Slice_3'^optimize/gradients/concat_grad/Slice_4

7optimize/gradients/concat_grad/tuple/control_dependencyIdentity$optimize/gradients/concat_grad/Slice0^optimize/gradients/concat_grad/tuple/group_deps*
T0*7
_class-
+)loc:@optimize/gradients/concat_grad/Slice*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9optimize/gradients/concat_grad/tuple/control_dependency_1Identity&optimize/gradients/concat_grad/Slice_10^optimize/gradients/concat_grad/tuple/group_deps*
T0*9
_class/
-+loc:@optimize/gradients/concat_grad/Slice_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9optimize/gradients/concat_grad/tuple/control_dependency_2Identity&optimize/gradients/concat_grad/Slice_20^optimize/gradients/concat_grad/tuple/group_deps*9
_class/
-+loc:@optimize/gradients/concat_grad/Slice_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

9optimize/gradients/concat_grad/tuple/control_dependency_3Identity&optimize/gradients/concat_grad/Slice_30^optimize/gradients/concat_grad/tuple/group_deps*
T0*9
_class/
-+loc:@optimize/gradients/concat_grad/Slice_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9optimize/gradients/concat_grad/tuple/control_dependency_4Identity&optimize/gradients/concat_grad/Slice_40^optimize/gradients/concat_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*9
_class/
-+loc:@optimize/gradients/concat_grad/Slice_4
r
*optimize/gradients/cnn0/Squeeze_grad/ShapeShapecnn0/gmp*
T0*
out_type0*
_output_shapes
:
ĺ
,optimize/gradients/cnn0/Squeeze_grad/ReshapeReshape7optimize/gradients/concat_grad/tuple/control_dependency*optimize/gradients/cnn0/Squeeze_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
r
*optimize/gradients/cnn1/Squeeze_grad/ShapeShapecnn1/gmp*
_output_shapes
:*
T0*
out_type0
ç
,optimize/gradients/cnn1/Squeeze_grad/ReshapeReshape9optimize/gradients/concat_grad/tuple/control_dependency_1*optimize/gradients/cnn1/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
*optimize/gradients/cnn2/Squeeze_grad/ShapeShapecnn2/gmp*
T0*
out_type0*
_output_shapes
:
ç
,optimize/gradients/cnn2/Squeeze_grad/ReshapeReshape9optimize/gradients/concat_grad/tuple/control_dependency_2*optimize/gradients/cnn2/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
*optimize/gradients/cnn3/Squeeze_grad/ShapeShapecnn3/gmp*
T0*
out_type0*
_output_shapes
:
ç
,optimize/gradients/cnn3/Squeeze_grad/ReshapeReshape9optimize/gradients/concat_grad/tuple/control_dependency_3*optimize/gradients/cnn3/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
*optimize/gradients/cnn4/Squeeze_grad/ShapeShapecnn4/gmp*
T0*
out_type0*
_output_shapes
:
ç
,optimize/gradients/cnn4/Squeeze_grad/ReshapeReshape9optimize/gradients/concat_grad/tuple/control_dependency_4*optimize/gradients/cnn4/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

,optimize/gradients/cnn0/gmp_grad/MaxPoolGradMaxPoolGrad	cnn0/convcnn0/gmp,optimize/gradients/cnn0/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

,optimize/gradients/cnn1/gmp_grad/MaxPoolGradMaxPoolGrad	cnn1/convcnn1/gmp,optimize/gradients/cnn1/Squeeze_grad/Reshape*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

,optimize/gradients/cnn2/gmp_grad/MaxPoolGradMaxPoolGrad	cnn2/convcnn2/gmp,optimize/gradients/cnn2/Squeeze_grad/Reshape*
ksize	
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC*
strides


,optimize/gradients/cnn3/gmp_grad/MaxPoolGradMaxPoolGrad	cnn3/convcnn3/gmp,optimize/gradients/cnn3/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

,optimize/gradients/cnn4/gmp_grad/MaxPoolGradMaxPoolGrad	cnn4/convcnn4/gmp,optimize/gradients/cnn4/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize	
*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
*optimize/gradients/cnn0/conv_grad/ReluGradReluGrad,optimize/gradients/cnn0/gmp_grad/MaxPoolGrad	cnn0/conv*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
*optimize/gradients/cnn1/conv_grad/ReluGradReluGrad,optimize/gradients/cnn1/gmp_grad/MaxPoolGrad	cnn1/conv*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
*optimize/gradients/cnn2/conv_grad/ReluGradReluGrad,optimize/gradients/cnn2/gmp_grad/MaxPoolGrad	cnn2/conv*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
*optimize/gradients/cnn3/conv_grad/ReluGradReluGrad,optimize/gradients/cnn3/gmp_grad/MaxPoolGrad	cnn3/conv*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
*optimize/gradients/cnn4/conv_grad/ReluGradReluGrad,optimize/gradients/cnn4/gmp_grad/MaxPoolGrad	cnn4/conv*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
0optimize/gradients/cnn0/BiasAdd_grad/BiasAddGradBiasAddGrad*optimize/gradients/cnn0/conv_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

5optimize/gradients/cnn0/BiasAdd_grad/tuple/group_depsNoOp1^optimize/gradients/cnn0/BiasAdd_grad/BiasAddGrad+^optimize/gradients/cnn0/conv_grad/ReluGrad
¨
=optimize/gradients/cnn0/BiasAdd_grad/tuple/control_dependencyIdentity*optimize/gradients/cnn0/conv_grad/ReluGrad6^optimize/gradients/cnn0/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@optimize/gradients/cnn0/conv_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
?optimize/gradients/cnn0/BiasAdd_grad/tuple/control_dependency_1Identity0optimize/gradients/cnn0/BiasAdd_grad/BiasAddGrad6^optimize/gradients/cnn0/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimize/gradients/cnn0/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
¨
0optimize/gradients/cnn1/BiasAdd_grad/BiasAddGradBiasAddGrad*optimize/gradients/cnn1/conv_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

5optimize/gradients/cnn1/BiasAdd_grad/tuple/group_depsNoOp1^optimize/gradients/cnn1/BiasAdd_grad/BiasAddGrad+^optimize/gradients/cnn1/conv_grad/ReluGrad
¨
=optimize/gradients/cnn1/BiasAdd_grad/tuple/control_dependencyIdentity*optimize/gradients/cnn1/conv_grad/ReluGrad6^optimize/gradients/cnn1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@optimize/gradients/cnn1/conv_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
?optimize/gradients/cnn1/BiasAdd_grad/tuple/control_dependency_1Identity0optimize/gradients/cnn1/BiasAdd_grad/BiasAddGrad6^optimize/gradients/cnn1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*C
_class9
75loc:@optimize/gradients/cnn1/BiasAdd_grad/BiasAddGrad
¨
0optimize/gradients/cnn2/BiasAdd_grad/BiasAddGradBiasAddGrad*optimize/gradients/cnn2/conv_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC

5optimize/gradients/cnn2/BiasAdd_grad/tuple/group_depsNoOp1^optimize/gradients/cnn2/BiasAdd_grad/BiasAddGrad+^optimize/gradients/cnn2/conv_grad/ReluGrad
¨
=optimize/gradients/cnn2/BiasAdd_grad/tuple/control_dependencyIdentity*optimize/gradients/cnn2/conv_grad/ReluGrad6^optimize/gradients/cnn2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@optimize/gradients/cnn2/conv_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
?optimize/gradients/cnn2/BiasAdd_grad/tuple/control_dependency_1Identity0optimize/gradients/cnn2/BiasAdd_grad/BiasAddGrad6^optimize/gradients/cnn2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimize/gradients/cnn2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
¨
0optimize/gradients/cnn3/BiasAdd_grad/BiasAddGradBiasAddGrad*optimize/gradients/cnn3/conv_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

5optimize/gradients/cnn3/BiasAdd_grad/tuple/group_depsNoOp1^optimize/gradients/cnn3/BiasAdd_grad/BiasAddGrad+^optimize/gradients/cnn3/conv_grad/ReluGrad
¨
=optimize/gradients/cnn3/BiasAdd_grad/tuple/control_dependencyIdentity*optimize/gradients/cnn3/conv_grad/ReluGrad6^optimize/gradients/cnn3/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@optimize/gradients/cnn3/conv_grad/ReluGrad
 
?optimize/gradients/cnn3/BiasAdd_grad/tuple/control_dependency_1Identity0optimize/gradients/cnn3/BiasAdd_grad/BiasAddGrad6^optimize/gradients/cnn3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimize/gradients/cnn3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
¨
0optimize/gradients/cnn4/BiasAdd_grad/BiasAddGradBiasAddGrad*optimize/gradients/cnn4/conv_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

5optimize/gradients/cnn4/BiasAdd_grad/tuple/group_depsNoOp1^optimize/gradients/cnn4/BiasAdd_grad/BiasAddGrad+^optimize/gradients/cnn4/conv_grad/ReluGrad
¨
=optimize/gradients/cnn4/BiasAdd_grad/tuple/control_dependencyIdentity*optimize/gradients/cnn4/conv_grad/ReluGrad6^optimize/gradients/cnn4/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@optimize/gradients/cnn4/conv_grad/ReluGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
?optimize/gradients/cnn4/BiasAdd_grad/tuple/control_dependency_1Identity0optimize/gradients/cnn4/BiasAdd_grad/BiasAddGrad6^optimize/gradients/cnn4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimize/gradients/cnn4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

*optimize/gradients/cnn0/Conv2D_grad/ShapeNShapeN
ExpandDimscnn0/w/read* 
_output_shapes
::*
T0*
out_type0*
N
ń
7optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*optimize/gradients/cnn0/Conv2D_grad/ShapeNcnn0/w/read=optimize/gradients/cnn0/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
	dilations
*
T0*
strides
*
data_formatNHWC
ë
8optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
ExpandDims,optimize/gradients/cnn0/Conv2D_grad/ShapeN:1=optimize/gradients/cnn0/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:Ź*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ą
4optimize/gradients/cnn0/Conv2D_grad/tuple/group_depsNoOp9^optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropFilter8^optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropInput
Ŕ
<optimize/gradients/cnn0/Conv2D_grad/tuple/control_dependencyIdentity7optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropInput5^optimize/gradients/cnn0/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
ť
>optimize/gradients/cnn0/Conv2D_grad/tuple/control_dependency_1Identity8optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropFilter5^optimize/gradients/cnn0/Conv2D_grad/tuple/group_deps*(
_output_shapes
:Ź*
T0*K
_classA
?=loc:@optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropFilter

*optimize/gradients/cnn1/Conv2D_grad/ShapeNShapeN
ExpandDimscnn1/w/read*
T0*
out_type0*
N* 
_output_shapes
::
ń
7optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*optimize/gradients/cnn1/Conv2D_grad/ShapeNcnn1/w/read=optimize/gradients/cnn1/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ë
8optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
ExpandDims,optimize/gradients/cnn1/Conv2D_grad/ShapeN:1=optimize/gradients/cnn1/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:Ź*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ą
4optimize/gradients/cnn1/Conv2D_grad/tuple/group_depsNoOp9^optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropFilter8^optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropInput
Ŕ
<optimize/gradients/cnn1/Conv2D_grad/tuple/control_dependencyIdentity7optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropInput5^optimize/gradients/cnn1/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
T0
ť
>optimize/gradients/cnn1/Conv2D_grad/tuple/control_dependency_1Identity8optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropFilter5^optimize/gradients/cnn1/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@optimize/gradients/cnn1/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:Ź*
T0

*optimize/gradients/cnn2/Conv2D_grad/ShapeNShapeN
ExpandDimscnn2/w/read* 
_output_shapes
::*
T0*
out_type0*
N
ń
7optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*optimize/gradients/cnn2/Conv2D_grad/ShapeNcnn2/w/read=optimize/gradients/cnn2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
ë
8optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
ExpandDims,optimize/gradients/cnn2/Conv2D_grad/ShapeN:1=optimize/gradients/cnn2/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:Ź*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ą
4optimize/gradients/cnn2/Conv2D_grad/tuple/group_depsNoOp9^optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropFilter8^optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropInput
Ŕ
<optimize/gradients/cnn2/Conv2D_grad/tuple/control_dependencyIdentity7optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropInput5^optimize/gradients/cnn2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
ť
>optimize/gradients/cnn2/Conv2D_grad/tuple/control_dependency_1Identity8optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropFilter5^optimize/gradients/cnn2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@optimize/gradients/cnn2/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:Ź

*optimize/gradients/cnn3/Conv2D_grad/ShapeNShapeN
ExpandDimscnn3/w/read*
T0*
out_type0*
N* 
_output_shapes
::
ń
7optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*optimize/gradients/cnn3/Conv2D_grad/ShapeNcnn3/w/read=optimize/gradients/cnn3/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
	dilations
*
T0
ë
8optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
ExpandDims,optimize/gradients/cnn3/Conv2D_grad/ShapeN:1=optimize/gradients/cnn3/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:Ź*
	dilations
*
T0
ą
4optimize/gradients/cnn3/Conv2D_grad/tuple/group_depsNoOp9^optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropFilter8^optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropInput
Ŕ
<optimize/gradients/cnn3/Conv2D_grad/tuple/control_dependencyIdentity7optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropInput5^optimize/gradients/cnn3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
ť
>optimize/gradients/cnn3/Conv2D_grad/tuple/control_dependency_1Identity8optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropFilter5^optimize/gradients/cnn3/Conv2D_grad/tuple/group_deps*(
_output_shapes
:Ź*
T0*K
_classA
?=loc:@optimize/gradients/cnn3/Conv2D_grad/Conv2DBackpropFilter

*optimize/gradients/cnn4/Conv2D_grad/ShapeNShapeN
ExpandDimscnn4/w/read* 
_output_shapes
::*
T0*
out_type0*
N
ń
7optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*optimize/gradients/cnn4/Conv2D_grad/ShapeNcnn4/w/read=optimize/gradients/cnn4/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
ë
8optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
ExpandDims,optimize/gradients/cnn4/Conv2D_grad/ShapeN:1=optimize/gradients/cnn4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:Ź
ą
4optimize/gradients/cnn4/Conv2D_grad/tuple/group_depsNoOp9^optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropFilter8^optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropInput
Ŕ
<optimize/gradients/cnn4/Conv2D_grad/tuple/control_dependencyIdentity7optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropInput5^optimize/gradients/cnn4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
ť
>optimize/gradients/cnn4/Conv2D_grad/tuple/control_dependency_1Identity8optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropFilter5^optimize/gradients/cnn4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@optimize/gradients/cnn4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:Ź
č
optimize/gradients/AddN_1AddN<optimize/gradients/cnn0/Conv2D_grad/tuple/control_dependency<optimize/gradients/cnn1/Conv2D_grad/tuple/control_dependency<optimize/gradients/cnn2/Conv2D_grad/tuple/control_dependency<optimize/gradients/cnn3/Conv2D_grad/tuple/control_dependency<optimize/gradients/cnn4/Conv2D_grad/tuple/control_dependency*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
T0*J
_class@
><loc:@optimize/gradients/cnn0/Conv2D_grad/Conv2DBackpropInput
|
(optimize/gradients/ExpandDims_grad/ShapeShapeembedding_1/Identity*
T0*
out_type0*
_output_shapes
:
Ŕ
*optimize/gradients/ExpandDims_grad/ReshapeReshapeoptimize/gradients/AddN_1(optimize/gradients/ExpandDims_grad/Shape*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
T0*
Tshape0
Ż
)optimize/gradients/embedding_1_grad/ShapeConst"/device:CPU:0*
_output_shapes
:*
_class
loc:@embedding*%
valueB	" N      ,      *
dtype0	
Ď
+optimize/gradients/embedding_1_grad/ToInt32Cast)optimize/gradients/embedding_1_grad/Shape"/device:CPU:0*

SrcT0	*
_class
loc:@embedding*
Truncate( *
_output_shapes
:*

DstT0
j
(optimize/gradients/embedding_1_grad/SizeSizeinput_x*
T0*
out_type0*
_output_shapes
: 
t
2optimize/gradients/embedding_1_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ë
.optimize/gradients/embedding_1_grad/ExpandDims
ExpandDims(optimize/gradients/embedding_1_grad/Size2optimize/gradients/embedding_1_grad/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0

7optimize/gradients/embedding_1_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

9optimize/gradients/embedding_1_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

9optimize/gradients/embedding_1_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
1optimize/gradients/embedding_1_grad/strided_sliceStridedSlice+optimize/gradients/embedding_1_grad/ToInt327optimize/gradients/embedding_1_grad/strided_slice/stack9optimize/gradients/embedding_1_grad/strided_slice/stack_19optimize/gradients/embedding_1_grad/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
q
/optimize/gradients/embedding_1_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

*optimize/gradients/embedding_1_grad/concatConcatV2.optimize/gradients/embedding_1_grad/ExpandDims1optimize/gradients/embedding_1_grad/strided_slice/optimize/gradients/embedding_1_grad/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ď
+optimize/gradients/embedding_1_grad/ReshapeReshape*optimize/gradients/ExpandDims_grad/Reshape*optimize/gradients/embedding_1_grad/concat*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
T0
­
-optimize/gradients/embedding_1_grad/Reshape_1Reshapeinput_x.optimize/gradients/embedding_1_grad/ExpandDims*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

"optimize/beta1_power/initial_valueConst*
_output_shapes
: *
_class
loc:@cnn0/b*
valueB
 *fff?*
dtype0

optimize/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@cnn0/b*
	container 
Ä
optimize/beta1_power/AssignAssignoptimize/beta1_power"optimize/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: 
w
optimize/beta1_power/readIdentityoptimize/beta1_power*
_output_shapes
: *
T0*
_class
loc:@cnn0/b

"optimize/beta2_power/initial_valueConst*
_output_shapes
: *
_class
loc:@cnn0/b*
valueB
 *wž?*
dtype0

optimize/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@cnn0/b
Ä
optimize/beta2_power/AssignAssignoptimize/beta2_power"optimize/beta2_power/initial_value*
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: *
use_locking(
w
optimize/beta2_power/readIdentityoptimize/beta2_power*
T0*
_class
loc:@cnn0/b*
_output_shapes
: 
Ž
0embedding/Adam/Initializer/zeros/shape_as_tensorConst"/device:CPU:0*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB" N  ,  

&embedding/Adam/Initializer/zeros/ConstConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
í
 embedding/Adam/Initializer/zerosFill0embedding/Adam/Initializer/zeros/shape_as_tensor&embedding/Adam/Initializer/zeros/Const"/device:CPU:0*
T0*
_class
loc:@embedding*

index_type0*!
_output_shapes
: Ź
ľ
embedding/Adam
VariableV2"/device:CPU:0*!
_output_shapes
: Ź*
shared_name *
_class
loc:@embedding*
	container *
shape: Ź*
dtype0
Ó
embedding/Adam/AssignAssignembedding/Adam embedding/Adam/Initializer/zeros"/device:CPU:0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź*
use_locking(*
T0

embedding/Adam/readIdentityembedding/Adam"/device:CPU:0*!
_output_shapes
: Ź*
T0*
_class
loc:@embedding
°
2embedding/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:CPU:0*
_output_shapes
:*
_class
loc:@embedding*
valueB" N  ,  *
dtype0

(embedding/Adam_1/Initializer/zeros/ConstConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *    *
dtype0*
_output_shapes
: 
ó
"embedding/Adam_1/Initializer/zerosFill2embedding/Adam_1/Initializer/zeros/shape_as_tensor(embedding/Adam_1/Initializer/zeros/Const"/device:CPU:0*
T0*
_class
loc:@embedding*

index_type0*!
_output_shapes
: Ź
ˇ
embedding/Adam_1
VariableV2"/device:CPU:0*
shared_name *
_class
loc:@embedding*
	container *
shape: Ź*
dtype0*!
_output_shapes
: Ź
Ů
embedding/Adam_1/AssignAssignembedding/Adam_1"embedding/Adam_1/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź

embedding/Adam_1/readIdentityembedding/Adam_1"/device:CPU:0*
T0*
_class
loc:@embedding*!
_output_shapes
: Ź
Ą
-cnn0/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn0/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

#cnn0/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@cnn0/w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ů
cnn0/w/Adam/Initializer/zerosFill-cnn0/w/Adam/Initializer/zeros/shape_as_tensor#cnn0/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@cnn0/w*

index_type0*(
_output_shapes
:Ź
Ž
cnn0/w/Adam
VariableV2*
shape:Ź*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn0/w*
	container 
ż
cnn0/w/Adam/AssignAssigncnn0/w/Adamcnn0/w/Adam/Initializer/zeros*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn0/w
w
cnn0/w/Adam/readIdentitycnn0/w/Adam*(
_output_shapes
:Ź*
T0*
_class
loc:@cnn0/w
Ł
/cnn0/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn0/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

%cnn0/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@cnn0/w*
valueB
 *    *
dtype0*
_output_shapes
: 
ß
cnn0/w/Adam_1/Initializer/zerosFill/cnn0/w/Adam_1/Initializer/zeros/shape_as_tensor%cnn0/w/Adam_1/Initializer/zeros/Const*
_class
loc:@cnn0/w*

index_type0*(
_output_shapes
:Ź*
T0
°
cnn0/w/Adam_1
VariableV2*
shared_name *
_class
loc:@cnn0/w*
	container *
shape:Ź*
dtype0*(
_output_shapes
:Ź
Ĺ
cnn0/w/Adam_1/AssignAssigncnn0/w/Adam_1cnn0/w/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź
{
cnn0/w/Adam_1/readIdentitycnn0/w/Adam_1*
_class
loc:@cnn0/w*(
_output_shapes
:Ź*
T0

cnn0/b/Adam/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@cnn0/b*
valueB*    *
dtype0

cnn0/b/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@cnn0/b*
	container *
shape:
˛
cnn0/b/Adam/AssignAssigncnn0/b/Adamcnn0/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn0/b
j
cnn0/b/Adam/readIdentitycnn0/b/Adam*
_output_shapes	
:*
T0*
_class
loc:@cnn0/b

cnn0/b/Adam_1/Initializer/zerosConst*
_class
loc:@cnn0/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn0/b/Adam_1
VariableV2*
shared_name *
_class
loc:@cnn0/b*
	container *
shape:*
dtype0*
_output_shapes	
:
¸
cnn0/b/Adam_1/AssignAssigncnn0/b/Adam_1cnn0/b/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(
n
cnn0/b/Adam_1/readIdentitycnn0/b/Adam_1*
T0*
_class
loc:@cnn0/b*
_output_shapes	
:
Ą
-cnn1/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn1/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

#cnn1/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@cnn1/w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ů
cnn1/w/Adam/Initializer/zerosFill-cnn1/w/Adam/Initializer/zeros/shape_as_tensor#cnn1/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@cnn1/w*

index_type0*(
_output_shapes
:Ź
Ž
cnn1/w/Adam
VariableV2*
	container *
shape:Ź*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn1/w
ż
cnn1/w/Adam/AssignAssigncnn1/w/Adamcnn1/w/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź
w
cnn1/w/Adam/readIdentitycnn1/w/Adam*
T0*
_class
loc:@cnn1/w*(
_output_shapes
:Ź
Ł
/cnn1/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn1/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

%cnn1/w/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@cnn1/w*
valueB
 *    *
dtype0
ß
cnn1/w/Adam_1/Initializer/zerosFill/cnn1/w/Adam_1/Initializer/zeros/shape_as_tensor%cnn1/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@cnn1/w*

index_type0*(
_output_shapes
:Ź
°
cnn1/w/Adam_1
VariableV2*
	container *
shape:Ź*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn1/w
Ĺ
cnn1/w/Adam_1/AssignAssigncnn1/w/Adam_1cnn1/w/Adam_1/Initializer/zeros*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
{
cnn1/w/Adam_1/readIdentitycnn1/w/Adam_1*
_class
loc:@cnn1/w*(
_output_shapes
:Ź*
T0

cnn1/b/Adam/Initializer/zerosConst*
_class
loc:@cnn1/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn1/b/Adam
VariableV2*
shared_name *
_class
loc:@cnn1/b*
	container *
shape:*
dtype0*
_output_shapes	
:
˛
cnn1/b/Adam/AssignAssigncnn1/b/Adamcnn1/b/Adam/Initializer/zeros*
T0*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:*
use_locking(
j
cnn1/b/Adam/readIdentitycnn1/b/Adam*
_output_shapes	
:*
T0*
_class
loc:@cnn1/b

cnn1/b/Adam_1/Initializer/zerosConst*
_class
loc:@cnn1/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn1/b/Adam_1
VariableV2*
shared_name *
_class
loc:@cnn1/b*
	container *
shape:*
dtype0*
_output_shapes	
:
¸
cnn1/b/Adam_1/AssignAssigncnn1/b/Adam_1cnn1/b/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn1/b*
validate_shape(
n
cnn1/b/Adam_1/readIdentitycnn1/b/Adam_1*
T0*
_class
loc:@cnn1/b*
_output_shapes	
:
Ą
-cnn2/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn2/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

#cnn2/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@cnn2/w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ů
cnn2/w/Adam/Initializer/zerosFill-cnn2/w/Adam/Initializer/zeros/shape_as_tensor#cnn2/w/Adam/Initializer/zeros/Const*(
_output_shapes
:Ź*
T0*
_class
loc:@cnn2/w*

index_type0
Ž
cnn2/w/Adam
VariableV2*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn2/w*
	container *
shape:Ź*
dtype0
ż
cnn2/w/Adam/AssignAssigncnn2/w/Adamcnn2/w/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź
w
cnn2/w/Adam/readIdentitycnn2/w/Adam*
T0*
_class
loc:@cnn2/w*(
_output_shapes
:Ź
Ł
/cnn2/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn2/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

%cnn2/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@cnn2/w*
valueB
 *    *
dtype0*
_output_shapes
: 
ß
cnn2/w/Adam_1/Initializer/zerosFill/cnn2/w/Adam_1/Initializer/zeros/shape_as_tensor%cnn2/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@cnn2/w*

index_type0*(
_output_shapes
:Ź
°
cnn2/w/Adam_1
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn2/w*
	container *
shape:Ź
Ĺ
cnn2/w/Adam_1/AssignAssigncnn2/w/Adam_1cnn2/w/Adam_1/Initializer/zeros*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
{
cnn2/w/Adam_1/readIdentitycnn2/w/Adam_1*
_class
loc:@cnn2/w*(
_output_shapes
:Ź*
T0

cnn2/b/Adam/Initializer/zerosConst*
_class
loc:@cnn2/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn2/b/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@cnn2/b*
	container *
shape:
˛
cnn2/b/Adam/AssignAssigncnn2/b/Adamcnn2/b/Adam/Initializer/zeros*
_class
loc:@cnn2/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
j
cnn2/b/Adam/readIdentitycnn2/b/Adam*
T0*
_class
loc:@cnn2/b*
_output_shapes	
:

cnn2/b/Adam_1/Initializer/zerosConst*
_class
loc:@cnn2/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn2/b/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@cnn2/b*
	container *
shape:
¸
cnn2/b/Adam_1/AssignAssigncnn2/b/Adam_1cnn2/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@cnn2/b*
validate_shape(*
_output_shapes	
:
n
cnn2/b/Adam_1/readIdentitycnn2/b/Adam_1*
_output_shapes	
:*
T0*
_class
loc:@cnn2/b
Ą
-cnn3/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn3/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

#cnn3/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@cnn3/w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ů
cnn3/w/Adam/Initializer/zerosFill-cnn3/w/Adam/Initializer/zeros/shape_as_tensor#cnn3/w/Adam/Initializer/zeros/Const*(
_output_shapes
:Ź*
T0*
_class
loc:@cnn3/w*

index_type0
Ž
cnn3/w/Adam
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn3/w*
	container *
shape:Ź
ż
cnn3/w/Adam/AssignAssigncnn3/w/Adamcnn3/w/Adam/Initializer/zeros*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn3/w
w
cnn3/w/Adam/readIdentitycnn3/w/Adam*
T0*
_class
loc:@cnn3/w*(
_output_shapes
:Ź
Ł
/cnn3/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@cnn3/w*%
valueB"   ,        

%cnn3/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@cnn3/w*
valueB
 *    *
dtype0*
_output_shapes
: 
ß
cnn3/w/Adam_1/Initializer/zerosFill/cnn3/w/Adam_1/Initializer/zeros/shape_as_tensor%cnn3/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@cnn3/w*

index_type0*(
_output_shapes
:Ź
°
cnn3/w/Adam_1
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn3/w*
	container *
shape:Ź
Ĺ
cnn3/w/Adam_1/AssignAssigncnn3/w/Adam_1cnn3/w/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź
{
cnn3/w/Adam_1/readIdentitycnn3/w/Adam_1*
T0*
_class
loc:@cnn3/w*(
_output_shapes
:Ź

cnn3/b/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@cnn3/b*
valueB*    

cnn3/b/Adam
VariableV2*
shared_name *
_class
loc:@cnn3/b*
	container *
shape:*
dtype0*
_output_shapes	
:
˛
cnn3/b/Adam/AssignAssigncnn3/b/Adamcnn3/b/Adam/Initializer/zeros*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
j
cnn3/b/Adam/readIdentitycnn3/b/Adam*
_class
loc:@cnn3/b*
_output_shapes	
:*
T0

cnn3/b/Adam_1/Initializer/zerosConst*
_class
loc:@cnn3/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn3/b/Adam_1
VariableV2*
_class
loc:@cnn3/b*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
¸
cnn3/b/Adam_1/AssignAssigncnn3/b/Adam_1cnn3/b/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn3/b*
validate_shape(
n
cnn3/b/Adam_1/readIdentitycnn3/b/Adam_1*
_output_shapes	
:*
T0*
_class
loc:@cnn3/b
Ą
-cnn4/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@cnn4/w*%
valueB"   ,        *
dtype0*
_output_shapes
:

#cnn4/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@cnn4/w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ů
cnn4/w/Adam/Initializer/zerosFill-cnn4/w/Adam/Initializer/zeros/shape_as_tensor#cnn4/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@cnn4/w*

index_type0*(
_output_shapes
:Ź
Ž
cnn4/w/Adam
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn4/w*
	container *
shape:Ź
ż
cnn4/w/Adam/AssignAssigncnn4/w/Adamcnn4/w/Adam/Initializer/zeros*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn4/w
w
cnn4/w/Adam/readIdentitycnn4/w/Adam*
T0*
_class
loc:@cnn4/w*(
_output_shapes
:Ź
Ł
/cnn4/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@cnn4/w*%
valueB"   ,        

%cnn4/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@cnn4/w*
valueB
 *    *
dtype0*
_output_shapes
: 
ß
cnn4/w/Adam_1/Initializer/zerosFill/cnn4/w/Adam_1/Initializer/zeros/shape_as_tensor%cnn4/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@cnn4/w*

index_type0*(
_output_shapes
:Ź
°
cnn4/w/Adam_1
VariableV2*
dtype0*(
_output_shapes
:Ź*
shared_name *
_class
loc:@cnn4/w*
	container *
shape:Ź
Ĺ
cnn4/w/Adam_1/AssignAssigncnn4/w/Adam_1cnn4/w/Adam_1/Initializer/zeros*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(
{
cnn4/w/Adam_1/readIdentitycnn4/w/Adam_1*
T0*
_class
loc:@cnn4/w*(
_output_shapes
:Ź

cnn4/b/Adam/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@cnn4/b*
valueB*    *
dtype0

cnn4/b/Adam
VariableV2*
_class
loc:@cnn4/b*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
˛
cnn4/b/Adam/AssignAssigncnn4/b/Adamcnn4/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn4/b
j
cnn4/b/Adam/readIdentitycnn4/b/Adam*
T0*
_class
loc:@cnn4/b*
_output_shapes	
:

cnn4/b/Adam_1/Initializer/zerosConst*
_class
loc:@cnn4/b*
valueB*    *
dtype0*
_output_shapes	
:

cnn4/b/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *
_class
loc:@cnn4/b*
	container *
shape:*
dtype0
¸
cnn4/b/Adam_1/AssignAssigncnn4/b/Adam_1cnn4/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(*
_output_shapes	
:
n
cnn4/b/Adam_1/readIdentitycnn4/b/Adam_1*
T0*
_class
loc:@cnn4/b*
_output_shapes	
:
Ľ
3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

)dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *    *
dtype0
é
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*

index_type0* 
_output_shapes
:


Ş
dense/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:

*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:


Ď
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(* 
_output_shapes
:



dense/kernel/Adam/readIdentitydense/kernel/Adam*
_class
loc:@dense/kernel* 
_output_shapes
:

*
T0
§
5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

+dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ď
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:

*
T0*
_class
loc:@dense/kernel*

index_type0
Ź
dense/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:

*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:


Ő
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(* 
_output_shapes
:



dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1* 
_output_shapes
:

*
T0*
_class
loc:@dense/kernel

!dense/bias/Adam/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@dense/bias*
	container *
shape:
Â
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
v
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_class
loc:@dense/bias*
_output_shapes	
:

#dense/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@dense/bias
Č
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:
z
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes	
:

(w/Adam/Initializer/zeros/shape_as_tensorConst*
_class

loc:@w*
valueB"   ¤  *
dtype0*
_output_shapes
:
y
w/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class

loc:@w*
valueB
 *    
˝
w/Adam/Initializer/zerosFill(w/Adam/Initializer/zeros/shape_as_tensorw/Adam/Initializer/zeros/Const*
T0*
_class

loc:@w*

index_type0* 
_output_shapes
:
¤

w/Adam
VariableV2*
_class

loc:@w*
	container *
shape:
¤*
dtype0* 
_output_shapes
:
¤*
shared_name 
Ł
w/Adam/AssignAssignw/Adamw/Adam/Initializer/zeros*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤*
use_locking(
`
w/Adam/readIdentityw/Adam* 
_output_shapes
:
¤*
T0*
_class

loc:@w

*w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class

loc:@w*
valueB"   ¤  *
dtype0
{
 w/Adam_1/Initializer/zeros/ConstConst*
_class

loc:@w*
valueB
 *    *
dtype0*
_output_shapes
: 
Ă
w/Adam_1/Initializer/zerosFill*w/Adam_1/Initializer/zeros/shape_as_tensor w/Adam_1/Initializer/zeros/Const*
T0*
_class

loc:@w*

index_type0* 
_output_shapes
:
¤

w/Adam_1
VariableV2*
shared_name *
_class

loc:@w*
	container *
shape:
¤*
dtype0* 
_output_shapes
:
¤
Š
w/Adam_1/AssignAssignw/Adam_1w/Adam_1/Initializer/zeros*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤*
use_locking(
d
w/Adam_1/readIdentityw/Adam_1*
T0*
_class

loc:@w* 
_output_shapes
:
¤

score/b/Adam/Initializer/zerosConst*
_class
loc:@score/b*
valueB¤*    *
dtype0*
_output_shapes	
:¤

score/b/Adam
VariableV2*
	container *
shape:¤*
dtype0*
_output_shapes	
:¤*
shared_name *
_class
loc:@score/b
ś
score/b/Adam/AssignAssignscore/b/Adamscore/b/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:¤*
use_locking(*
T0*
_class
loc:@score/b
m
score/b/Adam/readIdentityscore/b/Adam*
T0*
_class
loc:@score/b*
_output_shapes	
:¤

 score/b/Adam_1/Initializer/zerosConst*
_class
loc:@score/b*
valueB¤*    *
dtype0*
_output_shapes	
:¤

score/b/Adam_1
VariableV2*
	container *
shape:¤*
dtype0*
_output_shapes	
:¤*
shared_name *
_class
loc:@score/b
ź
score/b/Adam_1/AssignAssignscore/b/Adam_1 score/b/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@score/b*
validate_shape(*
_output_shapes	
:¤
q
score/b/Adam_1/readIdentityscore/b/Adam_1*
_output_shapes	
:¤*
T0*
_class
loc:@score/b
X
optimize/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
X
optimize/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Z
optimize/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
×
%optimize/Adam/update_embedding/UniqueUnique-optimize/gradients/embedding_1_grad/Reshape_1"/device:CPU:0*
out_idx0*
T0*
_class
loc:@embedding*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ś
$optimize/Adam/update_embedding/ShapeShape%optimize/Adam/update_embedding/Unique"/device:CPU:0*
_output_shapes
:*
T0*
_class
loc:@embedding*
out_type0
Š
2optimize/Adam/update_embedding/strided_slice/stackConst"/device:CPU:0*
_class
loc:@embedding*
valueB: *
dtype0*
_output_shapes
:
Ť
4optimize/Adam/update_embedding/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
_class
loc:@embedding*
valueB:
Ť
4optimize/Adam/update_embedding/strided_slice/stack_2Const"/device:CPU:0*
_class
loc:@embedding*
valueB:*
dtype0*
_output_shapes
:
Á
,optimize/Adam/update_embedding/strided_sliceStridedSlice$optimize/Adam/update_embedding/Shape2optimize/Adam/update_embedding/strided_slice/stack4optimize/Adam/update_embedding/strided_slice/stack_14optimize/Adam/update_embedding/strided_slice/stack_2"/device:CPU:0*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
_class
loc:@embedding*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
Ď
1optimize/Adam/update_embedding/UnsortedSegmentSumUnsortedSegmentSum+optimize/gradients/embedding_1_grad/Reshape'optimize/Adam/update_embedding/Unique:1,optimize/Adam/update_embedding/strided_slice"/device:CPU:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
Tnumsegments0*
Tindices0*
T0*
_class
loc:@embedding

$optimize/Adam/update_embedding/sub/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¸
"optimize/Adam/update_embedding/subSub$optimize/Adam/update_embedding/sub/xoptimize/beta2_power/read"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
: 

#optimize/Adam/update_embedding/SqrtSqrt"optimize/Adam/update_embedding/sub"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
: 
ˇ
"optimize/Adam/update_embedding/mulMuloptimize/ExponentialDecay#optimize/Adam/update_embedding/Sqrt"/device:CPU:0*
_class
loc:@embedding*
_output_shapes
: *
T0

&optimize/Adam/update_embedding/sub_1/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ź
$optimize/Adam/update_embedding/sub_1Sub&optimize/Adam/update_embedding/sub_1/xoptimize/beta1_power/read"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@embedding
É
&optimize/Adam/update_embedding/truedivRealDiv"optimize/Adam/update_embedding/mul$optimize/Adam/update_embedding/sub_1"/device:CPU:0*
T0*
_class
loc:@embedding*
_output_shapes
: 

&optimize/Adam/update_embedding/sub_2/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ś
$optimize/Adam/update_embedding/sub_2Sub&optimize/Adam/update_embedding/sub_2/xoptimize/Adam/beta1"/device:CPU:0*
_class
loc:@embedding*
_output_shapes
: *
T0
ä
$optimize/Adam/update_embedding/mul_1Mul1optimize/Adam/update_embedding/UnsortedSegmentSum$optimize/Adam/update_embedding/sub_2"/device:CPU:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
T0*
_class
loc:@embedding
Ž
$optimize/Adam/update_embedding/mul_2Mulembedding/Adam/readoptimize/Adam/beta1"/device:CPU:0*
T0*
_class
loc:@embedding*!
_output_shapes
: Ź
ç
%optimize/Adam/update_embedding/AssignAssignembedding/Adam$optimize/Adam/update_embedding/mul_2"/device:CPU:0*
use_locking( *
T0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź
¸
)optimize/Adam/update_embedding/ScatterAdd
ScatterAddembedding/Adam%optimize/Adam/update_embedding/Unique$optimize/Adam/update_embedding/mul_1&^optimize/Adam/update_embedding/Assign"/device:CPU:0*!
_output_shapes
: Ź*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding
ń
$optimize/Adam/update_embedding/mul_3Mul1optimize/Adam/update_embedding/UnsortedSegmentSum1optimize/Adam/update_embedding/UnsortedSegmentSum"/device:CPU:0*
T0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź

&optimize/Adam/update_embedding/sub_3/xConst"/device:CPU:0*
_class
loc:@embedding*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ś
$optimize/Adam/update_embedding/sub_3Sub&optimize/Adam/update_embedding/sub_3/xoptimize/Adam/beta2"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@embedding
×
$optimize/Adam/update_embedding/mul_4Mul$optimize/Adam/update_embedding/mul_3$optimize/Adam/update_embedding/sub_3"/device:CPU:0*
_class
loc:@embedding*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź*
T0
°
$optimize/Adam/update_embedding/mul_5Mulembedding/Adam_1/readoptimize/Adam/beta2"/device:CPU:0*
T0*
_class
loc:@embedding*!
_output_shapes
: Ź
ë
'optimize/Adam/update_embedding/Assign_1Assignembedding/Adam_1$optimize/Adam/update_embedding/mul_5"/device:CPU:0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź*
use_locking( *
T0
ž
+optimize/Adam/update_embedding/ScatterAdd_1
ScatterAddembedding/Adam_1%optimize/Adam/update_embedding/Unique$optimize/Adam/update_embedding/mul_4(^optimize/Adam/update_embedding/Assign_1"/device:CPU:0*!
_output_shapes
: Ź*
use_locking( *
Tindices0*
T0*
_class
loc:@embedding
ł
%optimize/Adam/update_embedding/Sqrt_1Sqrt+optimize/Adam/update_embedding/ScatterAdd_1"/device:CPU:0*
_class
loc:@embedding*!
_output_shapes
: Ź*
T0
×
$optimize/Adam/update_embedding/mul_6Mul&optimize/Adam/update_embedding/truediv)optimize/Adam/update_embedding/ScatterAdd"/device:CPU:0*
T0*
_class
loc:@embedding*!
_output_shapes
: Ź
Ŕ
"optimize/Adam/update_embedding/addAdd%optimize/Adam/update_embedding/Sqrt_1optimize/Adam/epsilon"/device:CPU:0*
T0*
_class
loc:@embedding*!
_output_shapes
: Ź
Ö
(optimize/Adam/update_embedding/truediv_1RealDiv$optimize/Adam/update_embedding/mul_6"optimize/Adam/update_embedding/add"/device:CPU:0*!
_output_shapes
: Ź*
T0*
_class
loc:@embedding
Ö
(optimize/Adam/update_embedding/AssignSub	AssignSub	embedding(optimize/Adam/update_embedding/truediv_1"/device:CPU:0*
use_locking( *
T0*
_class
loc:@embedding*!
_output_shapes
: Ź
ă
)optimize/Adam/update_embedding/group_depsNoOp)^optimize/Adam/update_embedding/AssignSub*^optimize/Adam/update_embedding/ScatterAdd,^optimize/Adam/update_embedding/ScatterAdd_1"/device:CPU:0*
_class
loc:@embedding

%optimize/Adam/update_cnn0/w/ApplyAdam	ApplyAdamcnn0/wcnn0/w/Adamcnn0/w/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon>optimize/gradients/cnn0/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@cnn0/w*
use_nesterov( *(
_output_shapes
:Ź

%optimize/Adam/update_cnn0/b/ApplyAdam	ApplyAdamcnn0/bcnn0/b/Adamcnn0/b/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon?optimize/gradients/cnn0/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@cnn0/b

%optimize/Adam/update_cnn1/w/ApplyAdam	ApplyAdamcnn1/wcnn1/w/Adamcnn1/w/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon>optimize/gradients/cnn1/Conv2D_grad/tuple/control_dependency_1*(
_output_shapes
:Ź*
use_locking( *
T0*
_class
loc:@cnn1/w*
use_nesterov( 

%optimize/Adam/update_cnn1/b/ApplyAdam	ApplyAdamcnn1/bcnn1/b/Adamcnn1/b/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon?optimize/gradients/cnn1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@cnn1/b*
use_nesterov( *
_output_shapes	
:*
use_locking( 

%optimize/Adam/update_cnn2/w/ApplyAdam	ApplyAdamcnn2/wcnn2/w/Adamcnn2/w/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon>optimize/gradients/cnn2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@cnn2/w*
use_nesterov( *(
_output_shapes
:Ź

%optimize/Adam/update_cnn2/b/ApplyAdam	ApplyAdamcnn2/bcnn2/b/Adamcnn2/b/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon?optimize/gradients/cnn2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@cnn2/b*
use_nesterov( *
_output_shapes	
:

%optimize/Adam/update_cnn3/w/ApplyAdam	ApplyAdamcnn3/wcnn3/w/Adamcnn3/w/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon>optimize/gradients/cnn3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@cnn3/w*
use_nesterov( *(
_output_shapes
:Ź

%optimize/Adam/update_cnn3/b/ApplyAdam	ApplyAdamcnn3/bcnn3/b/Adamcnn3/b/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon?optimize/gradients/cnn3/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@cnn3/b*
use_nesterov( *
_output_shapes	
:*
use_locking( 

%optimize/Adam/update_cnn4/w/ApplyAdam	ApplyAdamcnn4/wcnn4/w/Adamcnn4/w/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon>optimize/gradients/cnn4/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@cnn4/w*
use_nesterov( *(
_output_shapes
:Ź

%optimize/Adam/update_cnn4/b/ApplyAdam	ApplyAdamcnn4/bcnn4/b/Adamcnn4/b/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon?optimize/gradients/cnn4/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@cnn4/b
ş
+optimize/Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonEoptimize/gradients/score/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( * 
_output_shapes
:


Ź
)optimize/Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilonFoptimize/gradients/score/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
ý
 optimize/Adam/update_w/ApplyAdam	ApplyAdamww/Adamw/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon?optimize/gradients/score/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@w*
use_nesterov( * 
_output_shapes
:
¤

&optimize/Adam/update_score/b/ApplyAdam	ApplyAdamscore/bscore/b/Adamscore/b/Adam_1optimize/beta1_power/readoptimize/beta2_power/readoptimize/ExponentialDecayoptimize/Adam/beta1optimize/Adam/beta2optimize/Adam/epsilon@optimize/gradients/score/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@score/b*
use_nesterov( *
_output_shapes	
:¤
ć
optimize/Adam/mulMuloptimize/beta1_power/readoptimize/Adam/beta1&^optimize/Adam/update_cnn0/b/ApplyAdam&^optimize/Adam/update_cnn0/w/ApplyAdam&^optimize/Adam/update_cnn1/b/ApplyAdam&^optimize/Adam/update_cnn1/w/ApplyAdam&^optimize/Adam/update_cnn2/b/ApplyAdam&^optimize/Adam/update_cnn2/w/ApplyAdam&^optimize/Adam/update_cnn3/b/ApplyAdam&^optimize/Adam/update_cnn3/w/ApplyAdam&^optimize/Adam/update_cnn4/b/ApplyAdam&^optimize/Adam/update_cnn4/w/ApplyAdam*^optimize/Adam/update_dense/bias/ApplyAdam,^optimize/Adam/update_dense/kernel/ApplyAdam*^optimize/Adam/update_embedding/group_deps'^optimize/Adam/update_score/b/ApplyAdam!^optimize/Adam/update_w/ApplyAdam*
T0*
_class
loc:@cnn0/b*
_output_shapes
: 
Ź
optimize/Adam/AssignAssignoptimize/beta1_poweroptimize/Adam/mul*
use_locking( *
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: 
č
optimize/Adam/mul_1Muloptimize/beta2_power/readoptimize/Adam/beta2&^optimize/Adam/update_cnn0/b/ApplyAdam&^optimize/Adam/update_cnn0/w/ApplyAdam&^optimize/Adam/update_cnn1/b/ApplyAdam&^optimize/Adam/update_cnn1/w/ApplyAdam&^optimize/Adam/update_cnn2/b/ApplyAdam&^optimize/Adam/update_cnn2/w/ApplyAdam&^optimize/Adam/update_cnn3/b/ApplyAdam&^optimize/Adam/update_cnn3/w/ApplyAdam&^optimize/Adam/update_cnn4/b/ApplyAdam&^optimize/Adam/update_cnn4/w/ApplyAdam*^optimize/Adam/update_dense/bias/ApplyAdam,^optimize/Adam/update_dense/kernel/ApplyAdam*^optimize/Adam/update_embedding/group_deps'^optimize/Adam/update_score/b/ApplyAdam!^optimize/Adam/update_w/ApplyAdam*
T0*
_class
loc:@cnn0/b*
_output_shapes
: 
°
optimize/Adam/Assign_1Assignoptimize/beta2_poweroptimize/Adam/mul_1*
use_locking( *
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: 

optimize/Adam/NoOpNoOp^optimize/Adam/Assign^optimize/Adam/Assign_1&^optimize/Adam/update_cnn0/b/ApplyAdam&^optimize/Adam/update_cnn0/w/ApplyAdam&^optimize/Adam/update_cnn1/b/ApplyAdam&^optimize/Adam/update_cnn1/w/ApplyAdam&^optimize/Adam/update_cnn2/b/ApplyAdam&^optimize/Adam/update_cnn2/w/ApplyAdam&^optimize/Adam/update_cnn3/b/ApplyAdam&^optimize/Adam/update_cnn3/w/ApplyAdam&^optimize/Adam/update_cnn4/b/ApplyAdam&^optimize/Adam/update_cnn4/w/ApplyAdam*^optimize/Adam/update_dense/bias/ApplyAdam,^optimize/Adam/update_dense/kernel/ApplyAdam'^optimize/Adam/update_score/b/ApplyAdam!^optimize/Adam/update_w/ApplyAdam
W
optimize/Adam/NoOp_1NoOp*^optimize/Adam/update_embedding/group_deps"/device:CPU:0
A
optimize/AdamNoOp^optimize/Adam/NoOp^optimize/Adam/NoOp_1
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ř
save/SaveV2/tensor_namesConst*
valueBţ0Bcnn0/bBcnn0/b/AdamBcnn0/b/Adam_1Bcnn0/wBcnn0/w/AdamBcnn0/w/Adam_1Bcnn1/bBcnn1/b/AdamBcnn1/b/Adam_1Bcnn1/wBcnn1/w/AdamBcnn1/w/Adam_1Bcnn2/bBcnn2/b/AdamBcnn2/b/Adam_1Bcnn2/wBcnn2/w/AdamBcnn2/w/Adam_1Bcnn3/bBcnn3/b/AdamBcnn3/b/Adam_1Bcnn3/wBcnn3/w/AdamBcnn3/w/Adam_1Bcnn4/bBcnn4/b/AdamBcnn4/b/Adam_1Bcnn4/wBcnn4/w/AdamBcnn4/w/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1B	embeddingBembedding/AdamBembedding/Adam_1Boptimize/VariableBoptimize/beta1_powerBoptimize/beta2_powerBscore/bBscore/b/AdamBscore/b/Adam_1BwBw/AdamBw/Adam_1*
dtype0*
_output_shapes
:0
Ă
save/SaveV2/shape_and_slicesConst*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:0

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicescnn0/bcnn0/b/Adamcnn0/b/Adam_1cnn0/wcnn0/w/Adamcnn0/w/Adam_1cnn1/bcnn1/b/Adamcnn1/b/Adam_1cnn1/wcnn1/w/Adamcnn1/w/Adam_1cnn2/bcnn2/b/Adamcnn2/b/Adam_1cnn2/wcnn2/w/Adamcnn2/w/Adam_1cnn3/bcnn3/b/Adamcnn3/b/Adam_1cnn3/wcnn3/w/Adamcnn3/w/Adam_1cnn4/bcnn4/b/Adamcnn4/b/Adam_1cnn4/wcnn4/w/Adamcnn4/w/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1	embeddingembedding/Adamembedding/Adam_1optimize/Variableoptimize/beta1_poweroptimize/beta2_powerscore/bscore/b/Adamscore/b/Adam_1ww/Adamw/Adam_1*>
dtypes4
220
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
ę
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBţ0Bcnn0/bBcnn0/b/AdamBcnn0/b/Adam_1Bcnn0/wBcnn0/w/AdamBcnn0/w/Adam_1Bcnn1/bBcnn1/b/AdamBcnn1/b/Adam_1Bcnn1/wBcnn1/w/AdamBcnn1/w/Adam_1Bcnn2/bBcnn2/b/AdamBcnn2/b/Adam_1Bcnn2/wBcnn2/w/AdamBcnn2/w/Adam_1Bcnn3/bBcnn3/b/AdamBcnn3/b/Adam_1Bcnn3/wBcnn3/w/AdamBcnn3/w/Adam_1Bcnn4/bBcnn4/b/AdamBcnn4/b/Adam_1Bcnn4/wBcnn4/w/AdamBcnn4/w/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1B	embeddingBembedding/AdamBembedding/Adam_1Boptimize/VariableBoptimize/beta1_powerBoptimize/beta2_powerBscore/bBscore/b/AdamBscore/b/Adam_1BwBw/AdamBw/Adam_1*
dtype0*
_output_shapes
:0
Ő
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*>
dtypes4
220*Ö
_output_shapesĂ
Ŕ::::::::::::::::::::::::::::::::::::::::::::::::

save/AssignAssigncnn0/bsave/RestoreV2*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes	
:
 
save/Assign_1Assigncnn0/b/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes	
:
˘
save/Assign_2Assigncnn0/b/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes	
:
¨
save/Assign_3Assigncnn0/wsave/RestoreV2:3*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
­
save/Assign_4Assigncnn0/w/Adamsave/RestoreV2:4*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
Ż
save/Assign_5Assigncnn0/w/Adam_1save/RestoreV2:5*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0

save/Assign_6Assigncnn1/bsave/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn1/b
 
save/Assign_7Assigncnn1/b/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:
˘
save/Assign_8Assigncnn1/b/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:
¨
save/Assign_9Assigncnn1/wsave/RestoreV2:9*
use_locking(*
T0*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź
Ż
save/Assign_10Assigncnn1/w/Adamsave/RestoreV2:10*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
ą
save/Assign_11Assigncnn1/w/Adam_1save/RestoreV2:11*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0

save/Assign_12Assigncnn2/bsave/RestoreV2:12*
T0*
_class
loc:@cnn2/b*
validate_shape(*
_output_shapes	
:*
use_locking(
˘
save/Assign_13Assigncnn2/b/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@cnn2/b*
validate_shape(*
_output_shapes	
:
¤
save/Assign_14Assigncnn2/b/Adam_1save/RestoreV2:14*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn2/b
Ş
save/Assign_15Assigncnn2/wsave/RestoreV2:15*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn2/w
Ż
save/Assign_16Assigncnn2/w/Adamsave/RestoreV2:16*
T0*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(
ą
save/Assign_17Assigncnn2/w/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź

save/Assign_18Assigncnn3/bsave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:
˘
save/Assign_19Assigncnn3/b/Adamsave/RestoreV2:19*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
¤
save/Assign_20Assigncnn3/b/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_21Assigncnn3/wsave/RestoreV2:21*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
Ż
save/Assign_22Assigncnn3/w/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź
ą
save/Assign_23Assigncnn3/w/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź

save/Assign_24Assigncnn4/bsave/RestoreV2:24*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(
˘
save/Assign_25Assigncnn4/b/Adamsave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(*
_output_shapes	
:
¤
save/Assign_26Assigncnn4/b/Adam_1save/RestoreV2:26*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(
Ş
save/Assign_27Assigncnn4/wsave/RestoreV2:27*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0
Ż
save/Assign_28Assigncnn4/w/Adamsave/RestoreV2:28*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź
ą
save/Assign_29Assigncnn4/w/Adam_1save/RestoreV2:29*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź
Ľ
save/Assign_30Assign
dense/biassave/RestoreV2:30*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_31Assigndense/bias/Adamsave/RestoreV2:31*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ź
save/Assign_32Assigndense/bias/Adam_1save/RestoreV2:32*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ž
save/Assign_33Assigndense/kernelsave/RestoreV2:33* 
_output_shapes
:

*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
ł
save/Assign_34Assigndense/kernel/Adamsave/RestoreV2:34* 
_output_shapes
:

*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
ľ
save/Assign_35Assigndense/kernel/Adam_1save/RestoreV2:35*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(* 
_output_shapes
:


¸
save/Assign_36Assign	embeddingsave/RestoreV2:36"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź
˝
save/Assign_37Assignembedding/Adamsave/RestoreV2:37"/device:CPU:0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź*
use_locking(*
T0
ż
save/Assign_38Assignembedding/Adam_1save/RestoreV2:38"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź
Ž
save/Assign_39Assignoptimize/Variablesave/RestoreV2:39*
use_locking(*
T0*$
_class
loc:@optimize/Variable*
validate_shape(*
_output_shapes
: 
Ś
save/Assign_40Assignoptimize/beta1_powersave/RestoreV2:40*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(
Ś
save/Assign_41Assignoptimize/beta2_powersave/RestoreV2:41*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

save/Assign_42Assignscore/bsave/RestoreV2:42*
_class
loc:@score/b*
validate_shape(*
_output_shapes	
:¤*
use_locking(*
T0
¤
save/Assign_43Assignscore/b/Adamsave/RestoreV2:43*
use_locking(*
T0*
_class
loc:@score/b*
validate_shape(*
_output_shapes	
:¤
Ś
save/Assign_44Assignscore/b/Adam_1save/RestoreV2:44*
_output_shapes	
:¤*
use_locking(*
T0*
_class
loc:@score/b*
validate_shape(

save/Assign_45Assignwsave/RestoreV2:45*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤*
use_locking(

save/Assign_46Assignw/Adamsave/RestoreV2:46*
use_locking(*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤

save/Assign_47Assignw/Adam_1save/RestoreV2:47*
use_locking(*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤

save/restore_all/NoOpNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
a
save/restore_all/NoOp_1NoOp^save/Assign_36^save/Assign_37^save/Assign_38"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1
Â
	init/NoOpNoOp^cnn0/b/Adam/Assign^cnn0/b/Adam_1/Assign^cnn0/b/Assign^cnn0/w/Adam/Assign^cnn0/w/Adam_1/Assign^cnn0/w/Assign^cnn1/b/Adam/Assign^cnn1/b/Adam_1/Assign^cnn1/b/Assign^cnn1/w/Adam/Assign^cnn1/w/Adam_1/Assign^cnn1/w/Assign^cnn2/b/Adam/Assign^cnn2/b/Adam_1/Assign^cnn2/b/Assign^cnn2/w/Adam/Assign^cnn2/w/Adam_1/Assign^cnn2/w/Assign^cnn3/b/Adam/Assign^cnn3/b/Adam_1/Assign^cnn3/b/Assign^cnn3/w/Adam/Assign^cnn3/w/Adam_1/Assign^cnn3/w/Assign^cnn4/b/Adam/Assign^cnn4/b/Adam_1/Assign^cnn4/b/Assign^cnn4/w/Adam/Assign^cnn4/w/Adam_1/Assign^cnn4/w/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^optimize/Variable/Assign^optimize/beta1_power/Assign^optimize/beta2_power/Assign^score/b/Adam/Assign^score/b/Adam_1/Assign^score/b/Assign^w/Adam/Assign^w/Adam_1/Assign	^w/Assign
g
init/NoOp_1NoOp^embedding/Adam/Assign^embedding/Adam_1/Assign^embedding/Assign"/device:CPU:0
&
initNoOp
^init/NoOp^init/NoOp_1
R
save_1/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_9d1f2144367f4aba8160cd6b8ffda5c7/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
ź
save_1/SaveV2/tensor_namesConst"/device:CPU:0*Ţ
valueÔBŃ-Bcnn0/bBcnn0/b/AdamBcnn0/b/Adam_1Bcnn0/wBcnn0/w/AdamBcnn0/w/Adam_1Bcnn1/bBcnn1/b/AdamBcnn1/b/Adam_1Bcnn1/wBcnn1/w/AdamBcnn1/w/Adam_1Bcnn2/bBcnn2/b/AdamBcnn2/b/Adam_1Bcnn2/wBcnn2/w/AdamBcnn2/w/Adam_1Bcnn3/bBcnn3/b/AdamBcnn3/b/Adam_1Bcnn3/wBcnn3/w/AdamBcnn3/w/Adam_1Bcnn4/bBcnn4/b/AdamBcnn4/b/Adam_1Bcnn4/wBcnn4/w/AdamBcnn4/w/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Boptimize/VariableBoptimize/beta1_powerBoptimize/beta2_powerBscore/bBscore/b/AdamBscore/b/Adam_1BwBw/AdamBw/Adam_1*
dtype0*
_output_shapes
:-
Î
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-

save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicescnn0/bcnn0/b/Adamcnn0/b/Adam_1cnn0/wcnn0/w/Adamcnn0/w/Adam_1cnn1/bcnn1/b/Adamcnn1/b/Adam_1cnn1/wcnn1/w/Adamcnn1/w/Adam_1cnn2/bcnn2/b/Adamcnn2/b/Adam_1cnn2/wcnn2/w/Adamcnn2/w/Adam_1cnn3/bcnn3/b/Adamcnn3/b/Adam_1cnn3/wcnn3/w/Adamcnn3/w/Adam_1cnn4/bcnn4/b/Adamcnn4/b/Adam_1cnn4/wcnn4/w/Adamcnn4/w/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1optimize/Variableoptimize/beta1_poweroptimize/beta2_powerscore/bscore/b/Adamscore/b/Adam_1ww/Adamw/Adam_1"/device:CPU:0*;
dtypes1
/2-
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
o
save_1/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 

save_1/ShardedFilename_1ShardedFilenamesave_1/StringJoinsave_1/ShardedFilename_1/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 

save_1/SaveV2_1/tensor_namesConst"/device:CPU:0*@
value7B5B	embeddingBembedding/AdamBembedding/Adam_1*
dtype0*
_output_shapes
:
|
 save_1/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
Â
save_1/SaveV2_1SaveV2save_1/ShardedFilename_1save_1/SaveV2_1/tensor_names save_1/SaveV2_1/shape_and_slices	embeddingembedding/Adamembedding/Adam_1"/device:CPU:0*
dtypes
2
°
save_1/control_dependency_1Identitysave_1/ShardedFilename_1^save_1/SaveV2_1"/device:CPU:0*
_output_shapes
: *
T0*+
_class!
loc:@save_1/ShardedFilename_1
ę
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilenamesave_1/ShardedFilename_1^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
_output_shapes
:*
T0*

axis *
N

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
Ż
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0
ż
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*Ţ
valueÔBŃ-Bcnn0/bBcnn0/b/AdamBcnn0/b/Adam_1Bcnn0/wBcnn0/w/AdamBcnn0/w/Adam_1Bcnn1/bBcnn1/b/AdamBcnn1/b/Adam_1Bcnn1/wBcnn1/w/AdamBcnn1/w/Adam_1Bcnn2/bBcnn2/b/AdamBcnn2/b/Adam_1Bcnn2/wBcnn2/w/AdamBcnn2/w/Adam_1Bcnn3/bBcnn3/b/AdamBcnn3/b/Adam_1Bcnn3/wBcnn3/w/AdamBcnn3/w/Adam_1Bcnn4/bBcnn4/b/AdamBcnn4/b/Adam_1Bcnn4/wBcnn4/w/AdamBcnn4/w/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Boptimize/VariableBoptimize/beta1_powerBoptimize/beta2_powerBscore/bBscore/b/AdamBscore/b/Adam_1BwBw/AdamBw/Adam_1*
dtype0*
_output_shapes
:-
Ń
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-

save_1/AssignAssigncnn0/bsave_1/RestoreV2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn0/b
¤
save_1/Assign_1Assigncnn0/b/Adamsave_1/RestoreV2:1*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(
Ś
save_1/Assign_2Assigncnn0/b/Adam_1save_1/RestoreV2:2*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ź
save_1/Assign_3Assigncnn0/wsave_1/RestoreV2:3*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn0/w*
validate_shape(
ą
save_1/Assign_4Assigncnn0/w/Adamsave_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź
ł
save_1/Assign_5Assigncnn0/w/Adam_1save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@cnn0/w*
validate_shape(*(
_output_shapes
:Ź

save_1/Assign_6Assigncnn1/bsave_1/RestoreV2:6*
use_locking(*
T0*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:
¤
save_1/Assign_7Assigncnn1/b/Adamsave_1/RestoreV2:7*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ś
save_1/Assign_8Assigncnn1/b/Adam_1save_1/RestoreV2:8*
T0*
_class
loc:@cnn1/b*
validate_shape(*
_output_shapes	
:*
use_locking(
Ź
save_1/Assign_9Assigncnn1/wsave_1/RestoreV2:9*
T0*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(
ł
save_1/Assign_10Assigncnn1/w/Adamsave_1/RestoreV2:10*
use_locking(*
T0*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź
ľ
save_1/Assign_11Assigncnn1/w/Adam_1save_1/RestoreV2:11*
use_locking(*
T0*
_class
loc:@cnn1/w*
validate_shape(*(
_output_shapes
:Ź
Ą
save_1/Assign_12Assigncnn2/bsave_1/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn2/b
Ś
save_1/Assign_13Assigncnn2/b/Adamsave_1/RestoreV2:13*
use_locking(*
T0*
_class
loc:@cnn2/b*
validate_shape(*
_output_shapes	
:
¨
save_1/Assign_14Assigncnn2/b/Adam_1save_1/RestoreV2:14*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn2/b
Ž
save_1/Assign_15Assigncnn2/wsave_1/RestoreV2:15*
T0*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(
ł
save_1/Assign_16Assigncnn2/w/Adamsave_1/RestoreV2:16*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn2/w*
validate_shape(
ľ
save_1/Assign_17Assigncnn2/w/Adam_1save_1/RestoreV2:17*
use_locking(*
T0*
_class
loc:@cnn2/w*
validate_shape(*(
_output_shapes
:Ź
Ą
save_1/Assign_18Assigncnn3/bsave_1/RestoreV2:18*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@cnn3/b
Ś
save_1/Assign_19Assigncnn3/b/Adamsave_1/RestoreV2:19*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
¨
save_1/Assign_20Assigncnn3/b/Adam_1save_1/RestoreV2:20*
use_locking(*
T0*
_class
loc:@cnn3/b*
validate_shape(*
_output_shapes	
:
Ž
save_1/Assign_21Assigncnn3/wsave_1/RestoreV2:21*
use_locking(*
T0*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź
ł
save_1/Assign_22Assigncnn3/w/Adamsave_1/RestoreV2:22*
T0*
_class
loc:@cnn3/w*
validate_shape(*(
_output_shapes
:Ź*
use_locking(
ľ
save_1/Assign_23Assigncnn3/w/Adam_1save_1/RestoreV2:23*
validate_shape(*(
_output_shapes
:Ź*
use_locking(*
T0*
_class
loc:@cnn3/w
Ą
save_1/Assign_24Assigncnn4/bsave_1/RestoreV2:24*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(*
_output_shapes	
:
Ś
save_1/Assign_25Assigncnn4/b/Adamsave_1/RestoreV2:25*
use_locking(*
T0*
_class
loc:@cnn4/b*
validate_shape(*
_output_shapes	
:
¨
save_1/Assign_26Assigncnn4/b/Adam_1save_1/RestoreV2:26*
_class
loc:@cnn4/b*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ž
save_1/Assign_27Assigncnn4/wsave_1/RestoreV2:27*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź
ł
save_1/Assign_28Assigncnn4/w/Adamsave_1/RestoreV2:28*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź
ľ
save_1/Assign_29Assigncnn4/w/Adam_1save_1/RestoreV2:29*
use_locking(*
T0*
_class
loc:@cnn4/w*
validate_shape(*(
_output_shapes
:Ź
Š
save_1/Assign_30Assign
dense/biassave_1/RestoreV2:30*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:
Ž
save_1/Assign_31Assigndense/bias/Adamsave_1/RestoreV2:31*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
°
save_1/Assign_32Assigndense/bias/Adam_1save_1/RestoreV2:32*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˛
save_1/Assign_33Assigndense/kernelsave_1/RestoreV2:33*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(* 
_output_shapes
:


ˇ
save_1/Assign_34Assigndense/kernel/Adamsave_1/RestoreV2:34* 
_output_shapes
:

*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
š
save_1/Assign_35Assigndense/kernel/Adam_1save_1/RestoreV2:35*
T0*
_class
loc:@dense/kernel*
validate_shape(* 
_output_shapes
:

*
use_locking(
˛
save_1/Assign_36Assignoptimize/Variablesave_1/RestoreV2:36*$
_class
loc:@optimize/Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ş
save_1/Assign_37Assignoptimize/beta1_powersave_1/RestoreV2:37*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ş
save_1/Assign_38Assignoptimize/beta2_powersave_1/RestoreV2:38*
use_locking(*
T0*
_class
loc:@cnn0/b*
validate_shape(*
_output_shapes
: 
Ł
save_1/Assign_39Assignscore/bsave_1/RestoreV2:39*
use_locking(*
T0*
_class
loc:@score/b*
validate_shape(*
_output_shapes	
:¤
¨
save_1/Assign_40Assignscore/b/Adamsave_1/RestoreV2:40*
_output_shapes	
:¤*
use_locking(*
T0*
_class
loc:@score/b*
validate_shape(
Ş
save_1/Assign_41Assignscore/b/Adam_1save_1/RestoreV2:41*
use_locking(*
T0*
_class
loc:@score/b*
validate_shape(*
_output_shapes	
:¤

save_1/Assign_42Assignwsave_1/RestoreV2:42*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤*
use_locking(
Ą
save_1/Assign_43Assignw/Adamsave_1/RestoreV2:43*
T0*
_class

loc:@w*
validate_shape(* 
_output_shapes
:
¤*
use_locking(
Ł
save_1/Assign_44Assignw/Adam_1save_1/RestoreV2:44* 
_output_shapes
:
¤*
use_locking(*
T0*
_class

loc:@w*
validate_shape(
ç
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
˘
save_1/RestoreV2_1/tensor_namesConst"/device:CPU:0*@
value7B5B	embeddingBembedding/AdamBembedding/Adam_1*
dtype0*
_output_shapes
:

#save_1/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
ˇ
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2* 
_output_shapes
:::
ť
save_1/Assign_45Assign	embeddingsave_1/RestoreV2_1"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź
Â
save_1/Assign_46Assignembedding/Adamsave_1/RestoreV2_1:1"/device:CPU:0*!
_output_shapes
: Ź*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(
Ä
save_1/Assign_47Assignembedding/Adam_1save_1/RestoreV2_1:2"/device:CPU:0*
use_locking(*
T0*
_class
loc:@embedding*
validate_shape(*!
_output_shapes
: Ź
f
save_1/restore_shard_1NoOp^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47"/device:CPU:0
6
save_1/restore_all/NoOpNoOp^save_1/restore_shard
I
save_1/restore_all/NoOp_1NoOp^save_1/restore_shard_1"/device:CPU:0
P
save_1/restore_allNoOp^save_1/restore_all/NoOp^save_1/restore_all/NoOp_1"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"Ű	
trainable_variablesĂ	Ŕ	
[
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:08
O
cnn0/w:0cnn0/w/Assigncnn0/w/read:02#cnn0/w/Initializer/random_uniform:08
8
cnn0/b:0cnn0/b/Assigncnn0/b/read:02cnn0/Const:08
O
cnn1/w:0cnn1/w/Assigncnn1/w/read:02#cnn1/w/Initializer/random_uniform:08
8
cnn1/b:0cnn1/b/Assigncnn1/b/read:02cnn1/Const:08
O
cnn2/w:0cnn2/w/Assigncnn2/w/read:02#cnn2/w/Initializer/random_uniform:08
8
cnn2/b:0cnn2/b/Assigncnn2/b/read:02cnn2/Const:08
O
cnn3/w:0cnn3/w/Assigncnn3/w/read:02#cnn3/w/Initializer/random_uniform:08
8
cnn3/b:0cnn3/b/Assigncnn3/b/read:02cnn3/Const:08
O
cnn4/w:0cnn4/w/Assigncnn4/w/read:02#cnn4/w/Initializer/random_uniform:08
8
cnn4/b:0cnn4/b/Assigncnn4/b/read:02cnn4/Const:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
;
w:0w/Assignw/read:02w/Initializer/random_uniform:08
<
	score/b:0score/b/Assignscore/b/read:02score/Const:08
n
optimize/Variable:0optimize/Variable/Assignoptimize/Variable/read:02!optimize/Variable/initial_value:08"
train_op

optimize/Adam""
	variables""
[
embedding:0embedding/Assignembedding/read:02&embedding/Initializer/random_uniform:08
O
cnn0/w:0cnn0/w/Assigncnn0/w/read:02#cnn0/w/Initializer/random_uniform:08
8
cnn0/b:0cnn0/b/Assigncnn0/b/read:02cnn0/Const:08
O
cnn1/w:0cnn1/w/Assigncnn1/w/read:02#cnn1/w/Initializer/random_uniform:08
8
cnn1/b:0cnn1/b/Assigncnn1/b/read:02cnn1/Const:08
O
cnn2/w:0cnn2/w/Assigncnn2/w/read:02#cnn2/w/Initializer/random_uniform:08
8
cnn2/b:0cnn2/b/Assigncnn2/b/read:02cnn2/Const:08
O
cnn3/w:0cnn3/w/Assigncnn3/w/read:02#cnn3/w/Initializer/random_uniform:08
8
cnn3/b:0cnn3/b/Assigncnn3/b/read:02cnn3/Const:08
O
cnn4/w:0cnn4/w/Assigncnn4/w/read:02#cnn4/w/Initializer/random_uniform:08
8
cnn4/b:0cnn4/b/Assigncnn4/b/read:02cnn4/Const:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
;
w:0w/Assignw/read:02w/Initializer/random_uniform:08
<
	score/b:0score/b/Assignscore/b/read:02score/Const:08
n
optimize/Variable:0optimize/Variable/Assignoptimize/Variable/read:02!optimize/Variable/initial_value:08
x
optimize/beta1_power:0optimize/beta1_power/Assignoptimize/beta1_power/read:02$optimize/beta1_power/initial_value:0
x
optimize/beta2_power:0optimize/beta2_power/Assignoptimize/beta2_power/read:02$optimize/beta2_power/initial_value:0
d
embedding/Adam:0embedding/Adam/Assignembedding/Adam/read:02"embedding/Adam/Initializer/zeros:0
l
embedding/Adam_1:0embedding/Adam_1/Assignembedding/Adam_1/read:02$embedding/Adam_1/Initializer/zeros:0
X
cnn0/w/Adam:0cnn0/w/Adam/Assigncnn0/w/Adam/read:02cnn0/w/Adam/Initializer/zeros:0
`
cnn0/w/Adam_1:0cnn0/w/Adam_1/Assigncnn0/w/Adam_1/read:02!cnn0/w/Adam_1/Initializer/zeros:0
X
cnn0/b/Adam:0cnn0/b/Adam/Assigncnn0/b/Adam/read:02cnn0/b/Adam/Initializer/zeros:0
`
cnn0/b/Adam_1:0cnn0/b/Adam_1/Assigncnn0/b/Adam_1/read:02!cnn0/b/Adam_1/Initializer/zeros:0
X
cnn1/w/Adam:0cnn1/w/Adam/Assigncnn1/w/Adam/read:02cnn1/w/Adam/Initializer/zeros:0
`
cnn1/w/Adam_1:0cnn1/w/Adam_1/Assigncnn1/w/Adam_1/read:02!cnn1/w/Adam_1/Initializer/zeros:0
X
cnn1/b/Adam:0cnn1/b/Adam/Assigncnn1/b/Adam/read:02cnn1/b/Adam/Initializer/zeros:0
`
cnn1/b/Adam_1:0cnn1/b/Adam_1/Assigncnn1/b/Adam_1/read:02!cnn1/b/Adam_1/Initializer/zeros:0
X
cnn2/w/Adam:0cnn2/w/Adam/Assigncnn2/w/Adam/read:02cnn2/w/Adam/Initializer/zeros:0
`
cnn2/w/Adam_1:0cnn2/w/Adam_1/Assigncnn2/w/Adam_1/read:02!cnn2/w/Adam_1/Initializer/zeros:0
X
cnn2/b/Adam:0cnn2/b/Adam/Assigncnn2/b/Adam/read:02cnn2/b/Adam/Initializer/zeros:0
`
cnn2/b/Adam_1:0cnn2/b/Adam_1/Assigncnn2/b/Adam_1/read:02!cnn2/b/Adam_1/Initializer/zeros:0
X
cnn3/w/Adam:0cnn3/w/Adam/Assigncnn3/w/Adam/read:02cnn3/w/Adam/Initializer/zeros:0
`
cnn3/w/Adam_1:0cnn3/w/Adam_1/Assigncnn3/w/Adam_1/read:02!cnn3/w/Adam_1/Initializer/zeros:0
X
cnn3/b/Adam:0cnn3/b/Adam/Assigncnn3/b/Adam/read:02cnn3/b/Adam/Initializer/zeros:0
`
cnn3/b/Adam_1:0cnn3/b/Adam_1/Assigncnn3/b/Adam_1/read:02!cnn3/b/Adam_1/Initializer/zeros:0
X
cnn4/w/Adam:0cnn4/w/Adam/Assigncnn4/w/Adam/read:02cnn4/w/Adam/Initializer/zeros:0
`
cnn4/w/Adam_1:0cnn4/w/Adam_1/Assigncnn4/w/Adam_1/read:02!cnn4/w/Adam_1/Initializer/zeros:0
X
cnn4/b/Adam:0cnn4/b/Adam/Assigncnn4/b/Adam/read:02cnn4/b/Adam/Initializer/zeros:0
`
cnn4/b/Adam_1:0cnn4/b/Adam_1/Assigncnn4/b/Adam_1/read:02!cnn4/b/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
D
w/Adam:0w/Adam/Assignw/Adam/read:02w/Adam/Initializer/zeros:0
L

w/Adam_1:0w/Adam_1/Assignw/Adam_1/read:02w/Adam_1/Initializer/zeros:0
\
score/b/Adam:0score/b/Adam/Assignscore/b/Adam/read:02 score/b/Adam/Initializer/zeros:0
d
score/b/Adam_1:0score/b/Adam_1/Assignscore/b/Adam_1/read:02"score/b/Adam_1/Initializer/zeros:0*Ž
predict˘
 
	keep_prob
keep_prob:0
,
input_x!
	input_x:0˙˙˙˙˙˙˙˙˙4
	pred_prob'
score/Sigmoid:0˙˙˙˙˙˙˙˙˙¤tensorflow/serving/predict