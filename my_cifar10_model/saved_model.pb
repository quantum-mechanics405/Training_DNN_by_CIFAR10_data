��+
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��%
�
Nadam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_20/bias/v
{
)Nadam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_20/bias/v*
_output_shapes
:
*
dtype0
�
Nadam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*(
shared_nameNadam/dense_20/kernel/v
�
+Nadam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_20/kernel/v*
_output_shapes

:d
*
dtype0
�
Nadam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_19/bias/v
{
)Nadam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_19/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_19/kernel/v
�
+Nadam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_19/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_18/bias/v
{
)Nadam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_18/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_18/kernel/v
�
+Nadam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_18/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_17/bias/v
{
)Nadam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_17/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_17/kernel/v
�
+Nadam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_17/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_16/bias/v
{
)Nadam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_16/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_16/kernel/v
�
+Nadam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_16/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_15/bias/v
{
)Nadam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_15/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_15/kernel/v
�
+Nadam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_15/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_14/bias/v
{
)Nadam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_14/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_14/kernel/v
�
+Nadam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_14/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_13/bias/v
{
)Nadam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_13/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_13/kernel/v
�
+Nadam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_13/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_12/bias/v
{
)Nadam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_12/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_12/kernel/v
�
+Nadam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_12/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_11/bias/v
{
)Nadam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_11/kernel/v
�
+Nadam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_10/bias/v
{
)Nadam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_10/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_10/kernel/v
�
+Nadam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_10/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_9/bias/v
y
(Nadam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_9/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_9/kernel/v
�
*Nadam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_9/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_8/bias/v
y
(Nadam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_8/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_8/kernel/v
�
*Nadam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_8/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_7/bias/v
y
(Nadam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_7/kernel/v
�
*Nadam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_6/bias/v
y
(Nadam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_6/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_6/kernel/v
�
*Nadam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_6/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_5/bias/v
y
(Nadam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_5/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_5/kernel/v
�
*Nadam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_5/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_4/bias/v
y
(Nadam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_4/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_4/kernel/v
�
*Nadam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_4/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_3/bias/v
y
(Nadam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_3/kernel/v
�
*Nadam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_2/bias/v
y
(Nadam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_2/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_2/kernel/v
�
*Nadam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_2/kernel/v*
_output_shapes

:dd*
dtype0
�
Nadam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_1/bias/v
y
(Nadam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_1/kernel/v
�
*Nadam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/v*
_output_shapes

:dd*
dtype0
|
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameNadam/dense/bias/v
u
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes
:d*
dtype0
�
Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*%
shared_nameNadam/dense/kernel/v
~
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes
:	�d*
dtype0
�
Nadam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_20/bias/m
{
)Nadam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_20/bias/m*
_output_shapes
:
*
dtype0
�
Nadam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*(
shared_nameNadam/dense_20/kernel/m
�
+Nadam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_20/kernel/m*
_output_shapes

:d
*
dtype0
�
Nadam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_19/bias/m
{
)Nadam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_19/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_19/kernel/m
�
+Nadam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_19/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_18/bias/m
{
)Nadam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_18/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_18/kernel/m
�
+Nadam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_18/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_17/bias/m
{
)Nadam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_17/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_17/kernel/m
�
+Nadam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_17/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_16/bias/m
{
)Nadam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_16/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_16/kernel/m
�
+Nadam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_16/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_15/bias/m
{
)Nadam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_15/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_15/kernel/m
�
+Nadam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_15/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_14/bias/m
{
)Nadam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_14/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_14/kernel/m
�
+Nadam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_14/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_13/bias/m
{
)Nadam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_13/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_13/kernel/m
�
+Nadam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_13/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_12/bias/m
{
)Nadam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_12/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_12/kernel/m
�
+Nadam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_12/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_11/bias/m
{
)Nadam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_11/kernel/m
�
+Nadam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_10/bias/m
{
)Nadam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_10/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*(
shared_nameNadam/dense_10/kernel/m
�
+Nadam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_10/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_9/bias/m
y
(Nadam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_9/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_9/kernel/m
�
*Nadam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_9/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_8/bias/m
y
(Nadam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_8/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_8/kernel/m
�
*Nadam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_8/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_7/bias/m
y
(Nadam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_7/kernel/m
�
*Nadam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_6/bias/m
y
(Nadam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_6/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_6/kernel/m
�
*Nadam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_6/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_5/bias/m
y
(Nadam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_5/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_5/kernel/m
�
*Nadam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_5/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_4/bias/m
y
(Nadam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_4/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_4/kernel/m
�
*Nadam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_4/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_3/bias/m
y
(Nadam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_3/kernel/m
�
*Nadam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_2/bias/m
y
(Nadam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_2/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_2/kernel/m
�
*Nadam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_2/kernel/m*
_output_shapes

:dd*
dtype0
�
Nadam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameNadam/dense_1/bias/m
y
(Nadam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*'
shared_nameNadam/dense_1/kernel/m
�
*Nadam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/m*
_output_shapes

:dd*
dtype0
|
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameNadam/dense/bias/m
u
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes
:d*
dtype0
�
Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*%
shared_nameNadam/dense/kernel/m
~
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes
:	�d*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:
*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:d
*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:d*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:dd*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:d*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:dd*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:d*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:dd*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:d*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:dd*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:d*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:dd*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:d*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:dd*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:d*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:dd*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:d*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:dd*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:d*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:dd*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:d*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:dd*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:d*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:dd*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:d*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:dd*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:d*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:dd*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:d*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:dd*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:d*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:dd*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:d*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:dd*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:d*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:dd*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:d*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:dd*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:dd*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�d*
dtype0
�
serving_default_flatten_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_inputdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_103135

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
layer_with_weights-16
layer-17
layer_with_weights-17
layer-18
layer_with_weights-18
layer-19
layer_with_weights-19
layer-20
layer_with_weights-20
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias*
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias*
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias*
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias*
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
,0
-1
42
53
<4
=5
D6
E7
L8
M9
T10
U11
\12
]13
d14
e15
l16
m17
t18
u19
|20
}21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41*
�
,0
-1
42
53
<4
=5
D6
E7
L8
M9
T10
U11
\12
]13
d14
e15
l16
m17
t18
u19
|20
}21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate
�momentum_cache,m�-m�4m�5m�<m�=m�Dm�Em�Lm�Mm�Tm�Um�\m�]m�dm�em�lm�mm�tm�um�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�,v�-v�4v�5v�<v�=v�Dv�Ev�Lv�Mv�Tv�Uv�\v�]v�dv�ev�lv�mv�tv�uv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

,0
-1*

,0
-1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

d0
e1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

t0
u1*

t0
u1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_10/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_10/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_11/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_11/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_12/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_12/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_13/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_13/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_14/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_14/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_15/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_15/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_16/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_16/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_17/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_18/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_18/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_19/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_19/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_20/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_20/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�z
VARIABLE_VALUENadam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_8/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_8/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_9/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_9/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_10/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_10/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_11/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_11/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_12/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_12/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_13/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_13/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_14/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_14/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_15/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_15/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_16/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_16/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_17/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_17/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_18/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_18/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_19/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_19/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_20/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_20/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_8/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_8/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_9/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_9/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_10/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_10/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_11/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_11/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_12/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_12/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_13/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_13/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_14/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_14/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_15/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_15/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_16/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_16/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_17/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_17/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_18/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_18/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_19/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_19/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUENadam/dense_20/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUENadam/dense_20/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp*Nadam/dense_1/kernel/m/Read/ReadVariableOp(Nadam/dense_1/bias/m/Read/ReadVariableOp*Nadam/dense_2/kernel/m/Read/ReadVariableOp(Nadam/dense_2/bias/m/Read/ReadVariableOp*Nadam/dense_3/kernel/m/Read/ReadVariableOp(Nadam/dense_3/bias/m/Read/ReadVariableOp*Nadam/dense_4/kernel/m/Read/ReadVariableOp(Nadam/dense_4/bias/m/Read/ReadVariableOp*Nadam/dense_5/kernel/m/Read/ReadVariableOp(Nadam/dense_5/bias/m/Read/ReadVariableOp*Nadam/dense_6/kernel/m/Read/ReadVariableOp(Nadam/dense_6/bias/m/Read/ReadVariableOp*Nadam/dense_7/kernel/m/Read/ReadVariableOp(Nadam/dense_7/bias/m/Read/ReadVariableOp*Nadam/dense_8/kernel/m/Read/ReadVariableOp(Nadam/dense_8/bias/m/Read/ReadVariableOp*Nadam/dense_9/kernel/m/Read/ReadVariableOp(Nadam/dense_9/bias/m/Read/ReadVariableOp+Nadam/dense_10/kernel/m/Read/ReadVariableOp)Nadam/dense_10/bias/m/Read/ReadVariableOp+Nadam/dense_11/kernel/m/Read/ReadVariableOp)Nadam/dense_11/bias/m/Read/ReadVariableOp+Nadam/dense_12/kernel/m/Read/ReadVariableOp)Nadam/dense_12/bias/m/Read/ReadVariableOp+Nadam/dense_13/kernel/m/Read/ReadVariableOp)Nadam/dense_13/bias/m/Read/ReadVariableOp+Nadam/dense_14/kernel/m/Read/ReadVariableOp)Nadam/dense_14/bias/m/Read/ReadVariableOp+Nadam/dense_15/kernel/m/Read/ReadVariableOp)Nadam/dense_15/bias/m/Read/ReadVariableOp+Nadam/dense_16/kernel/m/Read/ReadVariableOp)Nadam/dense_16/bias/m/Read/ReadVariableOp+Nadam/dense_17/kernel/m/Read/ReadVariableOp)Nadam/dense_17/bias/m/Read/ReadVariableOp+Nadam/dense_18/kernel/m/Read/ReadVariableOp)Nadam/dense_18/bias/m/Read/ReadVariableOp+Nadam/dense_19/kernel/m/Read/ReadVariableOp)Nadam/dense_19/bias/m/Read/ReadVariableOp+Nadam/dense_20/kernel/m/Read/ReadVariableOp)Nadam/dense_20/bias/m/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp*Nadam/dense_1/kernel/v/Read/ReadVariableOp(Nadam/dense_1/bias/v/Read/ReadVariableOp*Nadam/dense_2/kernel/v/Read/ReadVariableOp(Nadam/dense_2/bias/v/Read/ReadVariableOp*Nadam/dense_3/kernel/v/Read/ReadVariableOp(Nadam/dense_3/bias/v/Read/ReadVariableOp*Nadam/dense_4/kernel/v/Read/ReadVariableOp(Nadam/dense_4/bias/v/Read/ReadVariableOp*Nadam/dense_5/kernel/v/Read/ReadVariableOp(Nadam/dense_5/bias/v/Read/ReadVariableOp*Nadam/dense_6/kernel/v/Read/ReadVariableOp(Nadam/dense_6/bias/v/Read/ReadVariableOp*Nadam/dense_7/kernel/v/Read/ReadVariableOp(Nadam/dense_7/bias/v/Read/ReadVariableOp*Nadam/dense_8/kernel/v/Read/ReadVariableOp(Nadam/dense_8/bias/v/Read/ReadVariableOp*Nadam/dense_9/kernel/v/Read/ReadVariableOp(Nadam/dense_9/bias/v/Read/ReadVariableOp+Nadam/dense_10/kernel/v/Read/ReadVariableOp)Nadam/dense_10/bias/v/Read/ReadVariableOp+Nadam/dense_11/kernel/v/Read/ReadVariableOp)Nadam/dense_11/bias/v/Read/ReadVariableOp+Nadam/dense_12/kernel/v/Read/ReadVariableOp)Nadam/dense_12/bias/v/Read/ReadVariableOp+Nadam/dense_13/kernel/v/Read/ReadVariableOp)Nadam/dense_13/bias/v/Read/ReadVariableOp+Nadam/dense_14/kernel/v/Read/ReadVariableOp)Nadam/dense_14/bias/v/Read/ReadVariableOp+Nadam/dense_15/kernel/v/Read/ReadVariableOp)Nadam/dense_15/bias/v/Read/ReadVariableOp+Nadam/dense_16/kernel/v/Read/ReadVariableOp)Nadam/dense_16/bias/v/Read/ReadVariableOp+Nadam/dense_17/kernel/v/Read/ReadVariableOp)Nadam/dense_17/bias/v/Read/ReadVariableOp+Nadam/dense_18/kernel/v/Read/ReadVariableOp)Nadam/dense_18/bias/v/Read/ReadVariableOp+Nadam/dense_19/kernel/v/Read/ReadVariableOp)Nadam/dense_19/bias/v/Read/ReadVariableOp+Nadam/dense_20/kernel/v/Read/ReadVariableOp)Nadam/dense_20/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_106701
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/dense/kernel/mNadam/dense/bias/mNadam/dense_1/kernel/mNadam/dense_1/bias/mNadam/dense_2/kernel/mNadam/dense_2/bias/mNadam/dense_3/kernel/mNadam/dense_3/bias/mNadam/dense_4/kernel/mNadam/dense_4/bias/mNadam/dense_5/kernel/mNadam/dense_5/bias/mNadam/dense_6/kernel/mNadam/dense_6/bias/mNadam/dense_7/kernel/mNadam/dense_7/bias/mNadam/dense_8/kernel/mNadam/dense_8/bias/mNadam/dense_9/kernel/mNadam/dense_9/bias/mNadam/dense_10/kernel/mNadam/dense_10/bias/mNadam/dense_11/kernel/mNadam/dense_11/bias/mNadam/dense_12/kernel/mNadam/dense_12/bias/mNadam/dense_13/kernel/mNadam/dense_13/bias/mNadam/dense_14/kernel/mNadam/dense_14/bias/mNadam/dense_15/kernel/mNadam/dense_15/bias/mNadam/dense_16/kernel/mNadam/dense_16/bias/mNadam/dense_17/kernel/mNadam/dense_17/bias/mNadam/dense_18/kernel/mNadam/dense_18/bias/mNadam/dense_19/kernel/mNadam/dense_19/bias/mNadam/dense_20/kernel/mNadam/dense_20/bias/mNadam/dense/kernel/vNadam/dense/bias/vNadam/dense_1/kernel/vNadam/dense_1/bias/vNadam/dense_2/kernel/vNadam/dense_2/bias/vNadam/dense_3/kernel/vNadam/dense_3/bias/vNadam/dense_4/kernel/vNadam/dense_4/bias/vNadam/dense_5/kernel/vNadam/dense_5/bias/vNadam/dense_6/kernel/vNadam/dense_6/bias/vNadam/dense_7/kernel/vNadam/dense_7/bias/vNadam/dense_8/kernel/vNadam/dense_8/bias/vNadam/dense_9/kernel/vNadam/dense_9/bias/vNadam/dense_10/kernel/vNadam/dense_10/bias/vNadam/dense_11/kernel/vNadam/dense_11/bias/vNadam/dense_12/kernel/vNadam/dense_12/bias/vNadam/dense_13/kernel/vNadam/dense_13/bias/vNadam/dense_14/kernel/vNadam/dense_14/bias/vNadam/dense_15/kernel/vNadam/dense_15/bias/vNadam/dense_16/kernel/vNadam/dense_16/bias/vNadam/dense_17/kernel/vNadam/dense_17/bias/vNadam/dense_18/kernel/vNadam/dense_18/bias/vNadam/dense_19/kernel/vNadam/dense_19/bias/vNadam/dense_20/kernel/vNadam/dense_20/bias/v*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_107119�� 
�
z
#__inference_internal_grad_fn_105447
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105465
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105123
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�j
�
F__inference_sequential_layer_call_and_return_conditional_losses_103038
flatten_input
dense_102932:	�d
dense_102934:d 
dense_1_102937:dd
dense_1_102939:d 
dense_2_102942:dd
dense_2_102944:d 
dense_3_102947:dd
dense_3_102949:d 
dense_4_102952:dd
dense_4_102954:d 
dense_5_102957:dd
dense_5_102959:d 
dense_6_102962:dd
dense_6_102964:d 
dense_7_102967:dd
dense_7_102969:d 
dense_8_102972:dd
dense_8_102974:d 
dense_9_102977:dd
dense_9_102979:d!
dense_10_102982:dd
dense_10_102984:d!
dense_11_102987:dd
dense_11_102989:d!
dense_12_102992:dd
dense_12_102994:d!
dense_13_102997:dd
dense_13_102999:d!
dense_14_103002:dd
dense_14_103004:d!
dense_15_103007:dd
dense_15_103009:d!
dense_16_103012:dd
dense_16_103014:d!
dense_17_103017:dd
dense_17_103019:d!
dense_18_103022:dd
dense_18_103024:d!
dense_19_103027:dd
dense_19_103029:d!
dense_20_103032:d

dense_20_103034:

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�dense_2/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101638�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_102932dense_102934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101658�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_102937dense_1_102939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101682�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_102942dense_2_102944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101706�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_102947dense_3_102949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_101730�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_102952dense_4_102954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_101754�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_102957dense_5_102959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_101778�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_102962dense_6_102964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_101802�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_102967dense_7_102969*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_101826�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_102972dense_8_102974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_101850�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_102977dense_9_102979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_101874�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_102982dense_10_102984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_101898�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_102987dense_11_102989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_101922�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_102992dense_12_102994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_101946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_102997dense_13_102999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_101970�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_103002dense_14_103004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_101994�
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_103007dense_15_103009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_102018�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_103012dense_16_103014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_102042�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_103017dense_17_103019*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_102066�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_103022dense_18_103024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_102090�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_103027dense_19_103029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_102114�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_103032dense_20_103034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_102131x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:^ Z
/
_output_shapes
:���������  
'
_user_specified_nameflatten_input
�
z
#__inference_internal_grad_fn_105375
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
)__inference_dense_16_layer_call_fn_104351

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_102042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105555
result_grads_0
result_grads_1
mul_dense_4_beta
mul_dense_4_biasadd
identityt
mulMulmul_dense_4_betamul_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_4_betamul_dense_4_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106041
result_grads_0
result_grads_1
mul_dense_11_beta
mul_dense_11_biasadd
identityv
mulMulmul_dense_11_betamul_dense_11_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_11_betamul_dense_11_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
D
(__inference_flatten_layer_call_fn_103904

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101638a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
D__inference_dense_14_layer_call_and_return_conditional_losses_101994

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101986*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_101638

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104781
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105627
result_grads_0
result_grads_1
mul_dense_8_beta
mul_dense_8_biasadd
identityt
mulMulmul_dense_8_betamul_dense_8_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_8_betamul_dense_8_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105969
result_grads_0
result_grads_1
mul_dense_7_beta
mul_dense_7_biasadd
identityt
mulMulmul_dense_7_betamul_dense_7_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_7_betamul_dense_7_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106221
result_grads_0
result_grads_1
mul_sequential_dense_1_beta"
mul_sequential_dense_1_biasadd
identity�
mulMulmul_sequential_dense_1_betamul_sequential_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_1_betamul_sequential_dense_1_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_104961
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_15_layer_call_and_return_conditional_losses_104342

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104334*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_9_layer_call_and_return_conditional_losses_101874

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101866*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105231
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105159
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
��
�%
!__inference__wrapped_model_101625
flatten_inputB
/sequential_dense_matmul_readvariableop_resource:	�d>
0sequential_dense_biasadd_readvariableop_resource:dC
1sequential_dense_1_matmul_readvariableop_resource:dd@
2sequential_dense_1_biasadd_readvariableop_resource:dC
1sequential_dense_2_matmul_readvariableop_resource:dd@
2sequential_dense_2_biasadd_readvariableop_resource:dC
1sequential_dense_3_matmul_readvariableop_resource:dd@
2sequential_dense_3_biasadd_readvariableop_resource:dC
1sequential_dense_4_matmul_readvariableop_resource:dd@
2sequential_dense_4_biasadd_readvariableop_resource:dC
1sequential_dense_5_matmul_readvariableop_resource:dd@
2sequential_dense_5_biasadd_readvariableop_resource:dC
1sequential_dense_6_matmul_readvariableop_resource:dd@
2sequential_dense_6_biasadd_readvariableop_resource:dC
1sequential_dense_7_matmul_readvariableop_resource:dd@
2sequential_dense_7_biasadd_readvariableop_resource:dC
1sequential_dense_8_matmul_readvariableop_resource:dd@
2sequential_dense_8_biasadd_readvariableop_resource:dC
1sequential_dense_9_matmul_readvariableop_resource:dd@
2sequential_dense_9_biasadd_readvariableop_resource:dD
2sequential_dense_10_matmul_readvariableop_resource:ddA
3sequential_dense_10_biasadd_readvariableop_resource:dD
2sequential_dense_11_matmul_readvariableop_resource:ddA
3sequential_dense_11_biasadd_readvariableop_resource:dD
2sequential_dense_12_matmul_readvariableop_resource:ddA
3sequential_dense_12_biasadd_readvariableop_resource:dD
2sequential_dense_13_matmul_readvariableop_resource:ddA
3sequential_dense_13_biasadd_readvariableop_resource:dD
2sequential_dense_14_matmul_readvariableop_resource:ddA
3sequential_dense_14_biasadd_readvariableop_resource:dD
2sequential_dense_15_matmul_readvariableop_resource:ddA
3sequential_dense_15_biasadd_readvariableop_resource:dD
2sequential_dense_16_matmul_readvariableop_resource:ddA
3sequential_dense_16_biasadd_readvariableop_resource:dD
2sequential_dense_17_matmul_readvariableop_resource:ddA
3sequential_dense_17_biasadd_readvariableop_resource:dD
2sequential_dense_18_matmul_readvariableop_resource:ddA
3sequential_dense_18_biasadd_readvariableop_resource:dD
2sequential_dense_19_matmul_readvariableop_resource:ddA
3sequential_dense_19_biasadd_readvariableop_resource:dD
2sequential_dense_20_matmul_readvariableop_resource:d
A
3sequential_dense_20_biasadd_readvariableop_resource:

identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�*sequential/dense_10/BiasAdd/ReadVariableOp�)sequential/dense_10/MatMul/ReadVariableOp�*sequential/dense_11/BiasAdd/ReadVariableOp�)sequential/dense_11/MatMul/ReadVariableOp�*sequential/dense_12/BiasAdd/ReadVariableOp�)sequential/dense_12/MatMul/ReadVariableOp�*sequential/dense_13/BiasAdd/ReadVariableOp�)sequential/dense_13/MatMul/ReadVariableOp�*sequential/dense_14/BiasAdd/ReadVariableOp�)sequential/dense_14/MatMul/ReadVariableOp�*sequential/dense_15/BiasAdd/ReadVariableOp�)sequential/dense_15/MatMul/ReadVariableOp�*sequential/dense_16/BiasAdd/ReadVariableOp�)sequential/dense_16/MatMul/ReadVariableOp�*sequential/dense_17/BiasAdd/ReadVariableOp�)sequential/dense_17/MatMul/ReadVariableOp�*sequential/dense_18/BiasAdd/ReadVariableOp�)sequential/dense_18/MatMul/ReadVariableOp�*sequential/dense_19/BiasAdd/ReadVariableOp�)sequential/dense_19/MatMul/ReadVariableOp�)sequential/dense_2/BiasAdd/ReadVariableOp�(sequential/dense_2/MatMul/ReadVariableOp�*sequential/dense_20/BiasAdd/ReadVariableOp�)sequential/dense_20/MatMul/ReadVariableOp�)sequential/dense_3/BiasAdd/ReadVariableOp�(sequential/dense_3/MatMul/ReadVariableOp�)sequential/dense_4/BiasAdd/ReadVariableOp�(sequential/dense_4/MatMul/ReadVariableOp�)sequential/dense_5/BiasAdd/ReadVariableOp�(sequential/dense_5/MatMul/ReadVariableOp�)sequential/dense_6/BiasAdd/ReadVariableOp�(sequential/dense_6/MatMul/ReadVariableOp�)sequential/dense_7/BiasAdd/ReadVariableOp�(sequential/dense_7/MatMul/ReadVariableOp�)sequential/dense_8/BiasAdd/ReadVariableOp�(sequential/dense_8/MatMul/ReadVariableOp�)sequential/dense_9/BiasAdd/ReadVariableOp�(sequential/dense_9/MatMul/ReadVariableOpi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential/flatten/ReshapeReshapeflatten_input!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dZ
sequential/dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense/mulMulsequential/dense/beta:output:0!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������do
sequential/dense/SigmoidSigmoidsequential/dense/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense/mul_1Mul!sequential/dense/BiasAdd:output:0sequential/dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������ds
sequential/dense/IdentityIdentitysequential/dense/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense/IdentityN	IdentityNsequential/dense/mul_1:z:0!sequential/dense/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101344*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_1/MatMulMatMul#sequential/dense/IdentityN:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_1/mulMul sequential/dense_1/beta:output:0#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_1/SigmoidSigmoidsequential/dense_1/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_1/mul_1Mul#sequential/dense_1/BiasAdd:output:0sequential/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_1/IdentityIdentitysequential/dense_1/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_1/IdentityN	IdentityNsequential/dense_1/mul_1:z:0#sequential/dense_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101358*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_2/MatMulMatMul%sequential/dense_1/IdentityN:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_2/mulMul sequential/dense_2/beta:output:0#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_2/SigmoidSigmoidsequential/dense_2/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_2/mul_1Mul#sequential/dense_2/BiasAdd:output:0sequential/dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_2/IdentityIdentitysequential/dense_2/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_2/IdentityN	IdentityNsequential/dense_2/mul_1:z:0#sequential/dense_2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101372*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_3/MatMulMatMul%sequential/dense_2/IdentityN:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_3/mulMul sequential/dense_3/beta:output:0#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_3/SigmoidSigmoidsequential/dense_3/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_3/mul_1Mul#sequential/dense_3/BiasAdd:output:0sequential/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_3/IdentityIdentitysequential/dense_3/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_3/IdentityN	IdentityNsequential/dense_3/mul_1:z:0#sequential/dense_3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101386*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_4/MatMul/ReadVariableOpReadVariableOp1sequential_dense_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_4/MatMulMatMul%sequential/dense_3/IdentityN:output:00sequential/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_4/BiasAddBiasAdd#sequential/dense_4/MatMul:product:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_4/mulMul sequential/dense_4/beta:output:0#sequential/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_4/SigmoidSigmoidsequential/dense_4/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_4/mul_1Mul#sequential/dense_4/BiasAdd:output:0sequential/dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_4/IdentityIdentitysequential/dense_4/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_4/IdentityN	IdentityNsequential/dense_4/mul_1:z:0#sequential/dense_4/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101400*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_5/MatMul/ReadVariableOpReadVariableOp1sequential_dense_5_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_5/MatMulMatMul%sequential/dense_4/IdentityN:output:00sequential/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_5/BiasAddBiasAdd#sequential/dense_5/MatMul:product:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_5/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_5/mulMul sequential/dense_5/beta:output:0#sequential/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_5/SigmoidSigmoidsequential/dense_5/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_5/mul_1Mul#sequential/dense_5/BiasAdd:output:0sequential/dense_5/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_5/IdentityIdentitysequential/dense_5/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_5/IdentityN	IdentityNsequential/dense_5/mul_1:z:0#sequential/dense_5/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101414*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_6/MatMul/ReadVariableOpReadVariableOp1sequential_dense_6_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_6/MatMulMatMul%sequential/dense_5/IdentityN:output:00sequential/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_6/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_6_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_6/BiasAddBiasAdd#sequential/dense_6/MatMul:product:01sequential/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_6/mulMul sequential/dense_6/beta:output:0#sequential/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_6/SigmoidSigmoidsequential/dense_6/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_6/mul_1Mul#sequential/dense_6/BiasAdd:output:0sequential/dense_6/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_6/IdentityIdentitysequential/dense_6/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_6/IdentityN	IdentityNsequential/dense_6/mul_1:z:0#sequential/dense_6/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101428*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_7/MatMul/ReadVariableOpReadVariableOp1sequential_dense_7_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_7/MatMulMatMul%sequential/dense_6/IdentityN:output:00sequential/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_7/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_7_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_7/BiasAddBiasAdd#sequential/dense_7/MatMul:product:01sequential/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_7/mulMul sequential/dense_7/beta:output:0#sequential/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_7/SigmoidSigmoidsequential/dense_7/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_7/mul_1Mul#sequential/dense_7/BiasAdd:output:0sequential/dense_7/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_7/IdentityIdentitysequential/dense_7/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_7/IdentityN	IdentityNsequential/dense_7/mul_1:z:0#sequential/dense_7/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101442*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_8/MatMul/ReadVariableOpReadVariableOp1sequential_dense_8_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_8/MatMulMatMul%sequential/dense_7/IdentityN:output:00sequential/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_8/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_8/BiasAddBiasAdd#sequential/dense_8/MatMul:product:01sequential/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_8/mulMul sequential/dense_8/beta:output:0#sequential/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_8/SigmoidSigmoidsequential/dense_8/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_8/mul_1Mul#sequential/dense_8/BiasAdd:output:0sequential/dense_8/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_8/IdentityIdentitysequential/dense_8/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_8/IdentityN	IdentityNsequential/dense_8/mul_1:z:0#sequential/dense_8/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101456*:
_output_shapes(
&:���������d:���������d�
(sequential/dense_9/MatMul/ReadVariableOpReadVariableOp1sequential_dense_9_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_9/MatMulMatMul%sequential/dense_8/IdentityN:output:00sequential/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_9/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_9_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_9/BiasAddBiasAdd#sequential/dense_9/MatMul:product:01sequential/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
sequential/dense_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_9/mulMul sequential/dense_9/beta:output:0#sequential/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������ds
sequential/dense_9/SigmoidSigmoidsequential/dense_9/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_9/mul_1Mul#sequential/dense_9/BiasAdd:output:0sequential/dense_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������dw
sequential/dense_9/IdentityIdentitysequential/dense_9/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_9/IdentityN	IdentityNsequential/dense_9/mul_1:z:0#sequential/dense_9/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101470*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_10/MatMul/ReadVariableOpReadVariableOp2sequential_dense_10_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_10/MatMulMatMul%sequential/dense_9/IdentityN:output:01sequential/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_10/BiasAddBiasAdd$sequential/dense_10/MatMul:product:02sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_10/mulMul!sequential/dense_10/beta:output:0$sequential/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_10/SigmoidSigmoidsequential/dense_10/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_10/mul_1Mul$sequential/dense_10/BiasAdd:output:0sequential/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_10/IdentityIdentitysequential/dense_10/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_10/IdentityN	IdentityNsequential/dense_10/mul_1:z:0$sequential/dense_10/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101484*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_11/MatMul/ReadVariableOpReadVariableOp2sequential_dense_11_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_11/MatMulMatMul&sequential/dense_10/IdentityN:output:01sequential/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_11/BiasAddBiasAdd$sequential/dense_11/MatMul:product:02sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_11/mulMul!sequential/dense_11/beta:output:0$sequential/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_11/SigmoidSigmoidsequential/dense_11/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_11/mul_1Mul$sequential/dense_11/BiasAdd:output:0sequential/dense_11/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_11/IdentityIdentitysequential/dense_11/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_11/IdentityN	IdentityNsequential/dense_11/mul_1:z:0$sequential/dense_11/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101498*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_12/MatMul/ReadVariableOpReadVariableOp2sequential_dense_12_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_12/MatMulMatMul&sequential/dense_11/IdentityN:output:01sequential/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_12/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_12_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_12/BiasAddBiasAdd$sequential/dense_12/MatMul:product:02sequential/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_12/mulMul!sequential/dense_12/beta:output:0$sequential/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_12/SigmoidSigmoidsequential/dense_12/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_12/mul_1Mul$sequential/dense_12/BiasAdd:output:0sequential/dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_12/IdentityIdentitysequential/dense_12/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_12/IdentityN	IdentityNsequential/dense_12/mul_1:z:0$sequential/dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101512*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_13/MatMul/ReadVariableOpReadVariableOp2sequential_dense_13_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_13/MatMulMatMul&sequential/dense_12/IdentityN:output:01sequential/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_13/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_13_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_13/BiasAddBiasAdd$sequential/dense_13/MatMul:product:02sequential/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_13/mulMul!sequential/dense_13/beta:output:0$sequential/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_13/SigmoidSigmoidsequential/dense_13/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_13/mul_1Mul$sequential/dense_13/BiasAdd:output:0sequential/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_13/IdentityIdentitysequential/dense_13/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_13/IdentityN	IdentityNsequential/dense_13/mul_1:z:0$sequential/dense_13/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101526*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_14/MatMul/ReadVariableOpReadVariableOp2sequential_dense_14_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_14/MatMulMatMul&sequential/dense_13/IdentityN:output:01sequential/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_14/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_14/BiasAddBiasAdd$sequential/dense_14/MatMul:product:02sequential/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_14/mulMul!sequential/dense_14/beta:output:0$sequential/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_14/SigmoidSigmoidsequential/dense_14/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_14/mul_1Mul$sequential/dense_14/BiasAdd:output:0sequential/dense_14/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_14/IdentityIdentitysequential/dense_14/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_14/IdentityN	IdentityNsequential/dense_14/mul_1:z:0$sequential/dense_14/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101540*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_15/MatMul/ReadVariableOpReadVariableOp2sequential_dense_15_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_15/MatMulMatMul&sequential/dense_14/IdentityN:output:01sequential/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_15/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_15_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_15/BiasAddBiasAdd$sequential/dense_15/MatMul:product:02sequential/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_15/mulMul!sequential/dense_15/beta:output:0$sequential/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_15/SigmoidSigmoidsequential/dense_15/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_15/mul_1Mul$sequential/dense_15/BiasAdd:output:0sequential/dense_15/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_15/IdentityIdentitysequential/dense_15/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_15/IdentityN	IdentityNsequential/dense_15/mul_1:z:0$sequential/dense_15/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101554*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_16/MatMul/ReadVariableOpReadVariableOp2sequential_dense_16_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_16/MatMulMatMul&sequential/dense_15/IdentityN:output:01sequential/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_16/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_16_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_16/BiasAddBiasAdd$sequential/dense_16/MatMul:product:02sequential/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_16/mulMul!sequential/dense_16/beta:output:0$sequential/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_16/SigmoidSigmoidsequential/dense_16/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_16/mul_1Mul$sequential/dense_16/BiasAdd:output:0sequential/dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_16/IdentityIdentitysequential/dense_16/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_16/IdentityN	IdentityNsequential/dense_16/mul_1:z:0$sequential/dense_16/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101568*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_17/MatMul/ReadVariableOpReadVariableOp2sequential_dense_17_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_17/MatMulMatMul&sequential/dense_16/IdentityN:output:01sequential/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_17/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_17_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_17/BiasAddBiasAdd$sequential/dense_17/MatMul:product:02sequential/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_17/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_17/mulMul!sequential/dense_17/beta:output:0$sequential/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_17/SigmoidSigmoidsequential/dense_17/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_17/mul_1Mul$sequential/dense_17/BiasAdd:output:0sequential/dense_17/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_17/IdentityIdentitysequential/dense_17/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_17/IdentityN	IdentityNsequential/dense_17/mul_1:z:0$sequential/dense_17/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101582*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_18/MatMul/ReadVariableOpReadVariableOp2sequential_dense_18_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_18/MatMulMatMul&sequential/dense_17/IdentityN:output:01sequential/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_18/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_18/BiasAddBiasAdd$sequential/dense_18/MatMul:product:02sequential/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_18/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_18/mulMul!sequential/dense_18/beta:output:0$sequential/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_18/SigmoidSigmoidsequential/dense_18/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_18/mul_1Mul$sequential/dense_18/BiasAdd:output:0sequential/dense_18/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_18/IdentityIdentitysequential/dense_18/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_18/IdentityN	IdentityNsequential/dense_18/mul_1:z:0$sequential/dense_18/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101596*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_19/MatMul/ReadVariableOpReadVariableOp2sequential_dense_19_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
sequential/dense_19/MatMulMatMul&sequential/dense_18/IdentityN:output:01sequential/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*sequential/dense_19/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_19_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_19/BiasAddBiasAdd$sequential/dense_19/MatMul:product:02sequential/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d]
sequential/dense_19/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/dense_19/mulMul!sequential/dense_19/beta:output:0$sequential/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������du
sequential/dense_19/SigmoidSigmoidsequential/dense_19/mul:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_19/mul_1Mul$sequential/dense_19/BiasAdd:output:0sequential/dense_19/Sigmoid:y:0*
T0*'
_output_shapes
:���������dy
sequential/dense_19/IdentityIdentitysequential/dense_19/mul_1:z:0*
T0*'
_output_shapes
:���������d�
sequential/dense_19/IdentityN	IdentityNsequential/dense_19/mul_1:z:0$sequential/dense_19/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101610*:
_output_shapes(
&:���������d:���������d�
)sequential/dense_20/MatMul/ReadVariableOpReadVariableOp2sequential_dense_20_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0�
sequential/dense_20/MatMulMatMul&sequential/dense_19/IdentityN:output:01sequential/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
*sequential/dense_20/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_20_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential/dense_20/BiasAddBiasAdd$sequential/dense_20/MatMul:product:02sequential/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
sequential/dense_20/SoftmaxSoftmax$sequential/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������
t
IdentityIdentity%sequential/dense_20/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp+^sequential/dense_10/BiasAdd/ReadVariableOp*^sequential/dense_10/MatMul/ReadVariableOp+^sequential/dense_11/BiasAdd/ReadVariableOp*^sequential/dense_11/MatMul/ReadVariableOp+^sequential/dense_12/BiasAdd/ReadVariableOp*^sequential/dense_12/MatMul/ReadVariableOp+^sequential/dense_13/BiasAdd/ReadVariableOp*^sequential/dense_13/MatMul/ReadVariableOp+^sequential/dense_14/BiasAdd/ReadVariableOp*^sequential/dense_14/MatMul/ReadVariableOp+^sequential/dense_15/BiasAdd/ReadVariableOp*^sequential/dense_15/MatMul/ReadVariableOp+^sequential/dense_16/BiasAdd/ReadVariableOp*^sequential/dense_16/MatMul/ReadVariableOp+^sequential/dense_17/BiasAdd/ReadVariableOp*^sequential/dense_17/MatMul/ReadVariableOp+^sequential/dense_18/BiasAdd/ReadVariableOp*^sequential/dense_18/MatMul/ReadVariableOp+^sequential/dense_19/BiasAdd/ReadVariableOp*^sequential/dense_19/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp+^sequential/dense_20/BiasAdd/ReadVariableOp*^sequential/dense_20/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp)^sequential/dense_4/MatMul/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp)^sequential/dense_5/MatMul/ReadVariableOp*^sequential/dense_6/BiasAdd/ReadVariableOp)^sequential/dense_6/MatMul/ReadVariableOp*^sequential/dense_7/BiasAdd/ReadVariableOp)^sequential/dense_7/MatMul/ReadVariableOp*^sequential/dense_8/BiasAdd/ReadVariableOp)^sequential/dense_8/MatMul/ReadVariableOp*^sequential/dense_9/BiasAdd/ReadVariableOp)^sequential/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2X
*sequential/dense_10/BiasAdd/ReadVariableOp*sequential/dense_10/BiasAdd/ReadVariableOp2V
)sequential/dense_10/MatMul/ReadVariableOp)sequential/dense_10/MatMul/ReadVariableOp2X
*sequential/dense_11/BiasAdd/ReadVariableOp*sequential/dense_11/BiasAdd/ReadVariableOp2V
)sequential/dense_11/MatMul/ReadVariableOp)sequential/dense_11/MatMul/ReadVariableOp2X
*sequential/dense_12/BiasAdd/ReadVariableOp*sequential/dense_12/BiasAdd/ReadVariableOp2V
)sequential/dense_12/MatMul/ReadVariableOp)sequential/dense_12/MatMul/ReadVariableOp2X
*sequential/dense_13/BiasAdd/ReadVariableOp*sequential/dense_13/BiasAdd/ReadVariableOp2V
)sequential/dense_13/MatMul/ReadVariableOp)sequential/dense_13/MatMul/ReadVariableOp2X
*sequential/dense_14/BiasAdd/ReadVariableOp*sequential/dense_14/BiasAdd/ReadVariableOp2V
)sequential/dense_14/MatMul/ReadVariableOp)sequential/dense_14/MatMul/ReadVariableOp2X
*sequential/dense_15/BiasAdd/ReadVariableOp*sequential/dense_15/BiasAdd/ReadVariableOp2V
)sequential/dense_15/MatMul/ReadVariableOp)sequential/dense_15/MatMul/ReadVariableOp2X
*sequential/dense_16/BiasAdd/ReadVariableOp*sequential/dense_16/BiasAdd/ReadVariableOp2V
)sequential/dense_16/MatMul/ReadVariableOp)sequential/dense_16/MatMul/ReadVariableOp2X
*sequential/dense_17/BiasAdd/ReadVariableOp*sequential/dense_17/BiasAdd/ReadVariableOp2V
)sequential/dense_17/MatMul/ReadVariableOp)sequential/dense_17/MatMul/ReadVariableOp2X
*sequential/dense_18/BiasAdd/ReadVariableOp*sequential/dense_18/BiasAdd/ReadVariableOp2V
)sequential/dense_18/MatMul/ReadVariableOp)sequential/dense_18/MatMul/ReadVariableOp2X
*sequential/dense_19/BiasAdd/ReadVariableOp*sequential/dense_19/BiasAdd/ReadVariableOp2V
)sequential/dense_19/MatMul/ReadVariableOp)sequential/dense_19/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2X
*sequential/dense_20/BiasAdd/ReadVariableOp*sequential/dense_20/BiasAdd/ReadVariableOp2V
)sequential/dense_20/MatMul/ReadVariableOp)sequential/dense_20/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2T
(sequential/dense_4/MatMul/ReadVariableOp(sequential/dense_4/MatMul/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2T
(sequential/dense_5/MatMul/ReadVariableOp(sequential/dense_5/MatMul/ReadVariableOp2V
)sequential/dense_6/BiasAdd/ReadVariableOp)sequential/dense_6/BiasAdd/ReadVariableOp2T
(sequential/dense_6/MatMul/ReadVariableOp(sequential/dense_6/MatMul/ReadVariableOp2V
)sequential/dense_7/BiasAdd/ReadVariableOp)sequential/dense_7/BiasAdd/ReadVariableOp2T
(sequential/dense_7/MatMul/ReadVariableOp(sequential/dense_7/MatMul/ReadVariableOp2V
)sequential/dense_8/BiasAdd/ReadVariableOp)sequential/dense_8/BiasAdd/ReadVariableOp2T
(sequential/dense_8/MatMul/ReadVariableOp(sequential/dense_8/MatMul/ReadVariableOp2V
)sequential/dense_9/BiasAdd/ReadVariableOp)sequential/dense_9/BiasAdd/ReadVariableOp2T
(sequential/dense_9/MatMul/ReadVariableOp(sequential/dense_9/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:���������  
'
_user_specified_nameflatten_input
�
z
#__inference_internal_grad_fn_105429
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105825
result_grads_0
result_grads_1
mul_dense_19_beta
mul_dense_19_biasadd
identityv
mulMulmul_dense_19_betamul_dense_19_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_19_betamul_dense_19_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�	
+__inference_sequential_layer_call_fn_103313

inputs
unknown:	�d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:dd

unknown_14:d

unknown_15:dd

unknown_16:d

unknown_17:dd

unknown_18:d

unknown_19:dd

unknown_20:d

unknown_21:dd

unknown_22:d

unknown_23:dd

unknown_24:d

unknown_25:dd

unknown_26:d

unknown_27:dd

unknown_28:d

unknown_29:dd

unknown_30:d

unknown_31:dd

unknown_32:d

unknown_33:dd

unknown_34:d

unknown_35:dd

unknown_36:d

unknown_37:dd

unknown_38:d

unknown_39:d


unknown_40:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_102642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_101778

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101770*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106275
result_grads_0
result_grads_1
mul_sequential_dense_4_beta"
mul_sequential_dense_4_biasadd
identity�
mulMulmul_sequential_dense_4_betamul_sequential_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_4_betamul_sequential_dense_4_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
A__inference_dense_layer_call_and_return_conditional_losses_103937

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103929*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104817
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105177
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_104835
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105537
result_grads_0
result_grads_1
mul_dense_3_beta
mul_dense_3_biasadd
identityt
mulMulmul_dense_3_betamul_dense_3_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_3_betamul_dense_3_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106095
result_grads_0
result_grads_1
mul_dense_14_beta
mul_dense_14_biasadd
identityv
mulMulmul_dense_14_betamul_dense_14_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_14_betamul_dense_14_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106077
result_grads_0
result_grads_1
mul_dense_13_beta
mul_dense_13_biasadd
identityv
mulMulmul_dense_13_betamul_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_13_betamul_dense_13_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
(__inference_dense_3_layer_call_fn_104000

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_101730o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_8_layer_call_and_return_conditional_losses_104153

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104145*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105915
result_grads_0
result_grads_1
mul_dense_4_beta
mul_dense_4_biasadd
identityt
mulMulmul_dense_4_betamul_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_4_betamul_dense_4_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105051
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105879
result_grads_0
result_grads_1
mul_dense_2_beta
mul_dense_2_biasadd
identityt
mulMulmul_dense_2_betamul_dense_2_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_2_betamul_dense_2_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
(__inference_dense_6_layer_call_fn_104081

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_101802o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
D__inference_dense_10_layer_call_and_return_conditional_losses_101898

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101890*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105735
result_grads_0
result_grads_1
mul_dense_14_beta
mul_dense_14_biasadd
identityv
mulMulmul_dense_14_betamul_dense_14_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_14_betamul_dense_14_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106437
result_grads_0
result_grads_1 
mul_sequential_dense_13_beta#
mul_sequential_dense_13_biasadd
identity�
mulMulmul_sequential_dense_13_betamul_sequential_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_13_betamul_sequential_dense_13_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_101826

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101818*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
D__inference_dense_12_layer_call_and_return_conditional_losses_101946

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101938*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_101706

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101698*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106149
result_grads_0
result_grads_1
mul_dense_17_beta
mul_dense_17_biasadd
identityv
mulMulmul_dense_17_betamul_dense_17_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_17_betamul_dense_17_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_11_layer_call_and_return_conditional_losses_101922

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101914*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105213
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105699
result_grads_0
result_grads_1
mul_dense_12_beta
mul_dense_12_biasadd
identityv
mulMulmul_dense_12_betamul_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_12_betamul_dense_12_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_104126

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104118*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106293
result_grads_0
result_grads_1
mul_sequential_dense_5_beta"
mul_sequential_dense_5_biasadd
identity�
mulMulmul_sequential_dense_5_betamul_sequential_dense_5_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_5_betamul_sequential_dense_5_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_17_layer_call_and_return_conditional_losses_104396

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104388*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
D__inference_dense_10_layer_call_and_return_conditional_losses_104207

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104199*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�	
+__inference_sequential_layer_call_fn_102225
flatten_input
unknown:	�d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:dd

unknown_14:d

unknown_15:dd

unknown_16:d

unknown_17:dd

unknown_18:d

unknown_19:dd

unknown_20:d

unknown_21:dd

unknown_22:d

unknown_23:dd

unknown_24:d

unknown_25:dd

unknown_26:d

unknown_27:dd

unknown_28:d

unknown_29:dd

unknown_30:d

unknown_31:dd

unknown_32:d

unknown_33:dd

unknown_34:d

unknown_35:dd

unknown_36:d

unknown_37:dd

unknown_38:d

unknown_39:d


unknown_40:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_102138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:���������  
'
_user_specified_nameflatten_input
�
�
(__inference_dense_5_layer_call_fn_104054

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_101778o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104871
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106239
result_grads_0
result_grads_1
mul_sequential_dense_2_beta"
mul_sequential_dense_2_biasadd
identity�
mulMulmul_sequential_dense_2_betamul_sequential_dense_2_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_2_betamul_sequential_dense_2_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_13_layer_call_and_return_conditional_losses_101970

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101962*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105357
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105951
result_grads_0
result_grads_1
mul_dense_6_beta
mul_dense_6_biasadd
identityt
mulMulmul_dense_6_betamul_dense_6_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_6_betamul_dense_6_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105321
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105195
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105501
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�

�
D__inference_dense_20_layer_call_and_return_conditional_losses_104470

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105987
result_grads_0
result_grads_1
mul_dense_8_beta
mul_dense_8_biasadd
identityt
mulMulmul_dense_8_betamul_dense_8_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_8_betamul_dense_8_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106185
result_grads_0
result_grads_1
mul_dense_19_beta
mul_dense_19_biasadd
identityv
mulMulmul_dense_19_betamul_dense_19_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_19_betamul_dense_19_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
(__inference_dense_4_layer_call_fn_104027

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_101754o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105609
result_grads_0
result_grads_1
mul_dense_7_beta
mul_dense_7_biasadd
identityt
mulMulmul_dense_7_betamul_dense_7_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_7_betamul_dense_7_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_104018

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104010*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105033
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106005
result_grads_0
result_grads_1
mul_dense_9_beta
mul_dense_9_biasadd
identityt
mulMulmul_dense_9_betamul_dense_9_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_9_betamul_dense_9_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_104943
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105789
result_grads_0
result_grads_1
mul_dense_17_beta
mul_dense_17_biasadd
identityv
mulMulmul_dense_17_betamul_dense_17_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_17_betamul_dense_17_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105807
result_grads_0
result_grads_1
mul_dense_18_beta
mul_dense_18_biasadd
identityv
mulMulmul_dense_18_betamul_dense_18_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_18_betamul_dense_18_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
(__inference_dense_1_layer_call_fn_103946

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106347
result_grads_0
result_grads_1
mul_sequential_dense_8_beta"
mul_sequential_dense_8_biasadd
identity�
mulMulmul_sequential_dense_8_betamul_sequential_dense_8_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_8_betamul_sequential_dense_8_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106527
result_grads_0
result_grads_1 
mul_sequential_dense_18_beta#
mul_sequential_dense_18_biasadd
identity�
mulMulmul_sequential_dense_18_betamul_sequential_dense_18_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_18_betamul_sequential_dense_18_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�	
+__inference_sequential_layer_call_fn_102818
flatten_input
unknown:	�d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:dd

unknown_14:d

unknown_15:dd

unknown_16:d

unknown_17:dd

unknown_18:d

unknown_19:dd

unknown_20:d

unknown_21:dd

unknown_22:d

unknown_23:dd

unknown_24:d

unknown_25:dd

unknown_26:d

unknown_27:dd

unknown_28:d

unknown_29:dd

unknown_30:d

unknown_31:dd

unknown_32:d

unknown_33:dd

unknown_34:d

unknown_35:dd

unknown_36:d

unknown_37:dd

unknown_38:d

unknown_39:d


unknown_40:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_102642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:���������  
'
_user_specified_nameflatten_input
�
�
#__inference_internal_grad_fn_106473
result_grads_0
result_grads_1 
mul_sequential_dense_15_beta#
mul_sequential_dense_15_biasadd
identity�
mulMulmul_sequential_dense_15_betamul_sequential_dense_15_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_15_betamul_sequential_dense_15_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_104763
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105087
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_104072

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104064*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
D__inference_dense_16_layer_call_and_return_conditional_losses_104369

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104361*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_101730

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101722*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
D__inference_dense_17_layer_call_and_return_conditional_losses_102066

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-102058*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106167
result_grads_0
result_grads_1
mul_dense_18_beta
mul_dense_18_biasadd
identityv
mulMulmul_dense_18_betamul_dense_18_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_18_betamul_dense_18_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106545
result_grads_0
result_grads_1 
mul_sequential_dense_19_beta#
mul_sequential_dense_19_biasadd
identity�
mulMulmul_sequential_dense_19_betamul_sequential_dense_19_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_19_betamul_sequential_dense_19_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106257
result_grads_0
result_grads_1
mul_sequential_dense_3_beta"
mul_sequential_dense_3_biasadd
identity�
mulMulmul_sequential_dense_3_betamul_sequential_dense_3_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_3_betamul_sequential_dense_3_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106509
result_grads_0
result_grads_1 
mul_sequential_dense_17_beta#
mul_sequential_dense_17_biasadd
identity�
mulMulmul_sequential_dense_17_betamul_sequential_dense_17_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_17_betamul_sequential_dense_17_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105285
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106401
result_grads_0
result_grads_1 
mul_sequential_dense_11_beta#
mul_sequential_dense_11_biasadd
identity�
mulMulmul_sequential_dense_11_betamul_sequential_dense_11_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_11_betamul_sequential_dense_11_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_12_layer_call_and_return_conditional_losses_104261

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104253*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106329
result_grads_0
result_grads_1
mul_sequential_dense_7_beta"
mul_sequential_dense_7_biasadd
identity�
mulMulmul_sequential_dense_7_betamul_sequential_dense_7_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_7_betamul_sequential_dense_7_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�

�
D__inference_dense_20_layer_call_and_return_conditional_losses_102131

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_13_layer_call_fn_104270

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_101970o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105015
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105249
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105483
result_grads_0
result_grads_1
mul_dense_beta
mul_dense_biasadd
identityp
mulMulmul_dense_betamul_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������da
mul_1Mulmul_dense_betamul_dense_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106311
result_grads_0
result_grads_1
mul_sequential_dense_6_beta"
mul_sequential_dense_6_biasadd
identity�
mulMulmul_sequential_dense_6_betamul_sequential_dense_6_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_6_betamul_sequential_dense_6_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
)__inference_dense_10_layer_call_fn_104189

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_101898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�	
$__inference_signature_wrapper_103135
flatten_input
unknown:	�d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:dd

unknown_14:d

unknown_15:dd

unknown_16:d

unknown_17:dd

unknown_18:d

unknown_19:dd

unknown_20:d

unknown_21:dd

unknown_22:d

unknown_23:dd

unknown_24:d

unknown_25:dd

unknown_26:d

unknown_27:dd

unknown_28:d

unknown_29:dd

unknown_30:d

unknown_31:dd

unknown_32:d

unknown_33:dd

unknown_34:d

unknown_35:dd

unknown_36:d

unknown_37:dd

unknown_38:d

unknown_39:d


unknown_40:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_101625o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:���������  
'
_user_specified_nameflatten_input
�
�
#__inference_internal_grad_fn_106383
result_grads_0
result_grads_1 
mul_sequential_dense_10_beta#
mul_sequential_dense_10_biasadd
identity�
mulMulmul_sequential_dense_10_betamul_sequential_dense_10_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_10_betamul_sequential_dense_10_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_15_layer_call_and_return_conditional_losses_102018

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-102010*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104889
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105591
result_grads_0
result_grads_1
mul_dense_6_beta
mul_dense_6_biasadd
identityt
mulMulmul_dense_6_betamul_dense_6_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_6_betamul_dense_6_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105861
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105933
result_grads_0
result_grads_1
mul_dense_5_beta
mul_dense_5_biasadd
identityt
mulMulmul_dense_5_betamul_dense_5_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_5_betamul_dense_5_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105843
result_grads_0
result_grads_1
mul_dense_beta
mul_dense_biasadd
identityp
mulMulmul_dense_betamul_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������da
mul_1Mulmul_dense_betamul_dense_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
A__inference_dense_layer_call_and_return_conditional_losses_101658

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101650*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_9_layer_call_fn_104162

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_101874o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105393
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105663
result_grads_0
result_grads_1
mul_dense_10_beta
mul_dense_10_biasadd
identityv
mulMulmul_dense_10_betamul_dense_10_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_10_betamul_dense_10_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
��
�
F__inference_sequential_layer_call_and_return_conditional_losses_103606

inputs7
$dense_matmul_readvariableop_resource:	�d3
%dense_biasadd_readvariableop_resource:d8
&dense_1_matmul_readvariableop_resource:dd5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:dd5
'dense_2_biasadd_readvariableop_resource:d8
&dense_3_matmul_readvariableop_resource:dd5
'dense_3_biasadd_readvariableop_resource:d8
&dense_4_matmul_readvariableop_resource:dd5
'dense_4_biasadd_readvariableop_resource:d8
&dense_5_matmul_readvariableop_resource:dd5
'dense_5_biasadd_readvariableop_resource:d8
&dense_6_matmul_readvariableop_resource:dd5
'dense_6_biasadd_readvariableop_resource:d8
&dense_7_matmul_readvariableop_resource:dd5
'dense_7_biasadd_readvariableop_resource:d8
&dense_8_matmul_readvariableop_resource:dd5
'dense_8_biasadd_readvariableop_resource:d8
&dense_9_matmul_readvariableop_resource:dd5
'dense_9_biasadd_readvariableop_resource:d9
'dense_10_matmul_readvariableop_resource:dd6
(dense_10_biasadd_readvariableop_resource:d9
'dense_11_matmul_readvariableop_resource:dd6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:dd6
(dense_12_biasadd_readvariableop_resource:d9
'dense_13_matmul_readvariableop_resource:dd6
(dense_13_biasadd_readvariableop_resource:d9
'dense_14_matmul_readvariableop_resource:dd6
(dense_14_biasadd_readvariableop_resource:d9
'dense_15_matmul_readvariableop_resource:dd6
(dense_15_biasadd_readvariableop_resource:d9
'dense_16_matmul_readvariableop_resource:dd6
(dense_16_biasadd_readvariableop_resource:d9
'dense_17_matmul_readvariableop_resource:dd6
(dense_17_biasadd_readvariableop_resource:d9
'dense_18_matmul_readvariableop_resource:dd6
(dense_18_biasadd_readvariableop_resource:d9
'dense_19_matmul_readvariableop_resource:dd6
(dense_19_biasadd_readvariableop_resource:d9
'dense_20_matmul_readvariableop_resource:d
6
(dense_20_biasadd_readvariableop_resource:

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dO

dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	dense/mulMuldense/beta:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������dY
dense/SigmoidSigmoiddense/mul:z:0*
T0*'
_output_shapes
:���������do
dense/mul_1Muldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������d]
dense/IdentityIdentitydense/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense/IdentityN	IdentityNdense/mul_1:z:0dense/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103325*:
_output_shapes(
&:���������d:���������d�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:���������du
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103339*:
_output_shapes(
&:���������d:���������d�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_2/MatMulMatMuldense_1/IdentityN:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_2/mulMuldense_2/beta:output:0dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_2/SigmoidSigmoiddense_2/mul:z:0*
T0*'
_output_shapes
:���������du
dense_2/mul_1Muldense_2/BiasAdd:output:0dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_2/IdentityIdentitydense_2/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_2/IdentityN	IdentityNdense_2/mul_1:z:0dense_2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103353*:
_output_shapes(
&:���������d:���������d�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_3/MatMulMatMuldense_2/IdentityN:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_3/mulMuldense_3/beta:output:0dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_3/SigmoidSigmoiddense_3/mul:z:0*
T0*'
_output_shapes
:���������du
dense_3/mul_1Muldense_3/BiasAdd:output:0dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_3/IdentityIdentitydense_3/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_3/IdentityN	IdentityNdense_3/mul_1:z:0dense_3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103367*:
_output_shapes(
&:���������d:���������d�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_4/MatMulMatMuldense_3/IdentityN:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_4/mulMuldense_4/beta:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_4/SigmoidSigmoiddense_4/mul:z:0*
T0*'
_output_shapes
:���������du
dense_4/mul_1Muldense_4/BiasAdd:output:0dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_4/IdentityIdentitydense_4/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_4/IdentityN	IdentityNdense_4/mul_1:z:0dense_4/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103381*:
_output_shapes(
&:���������d:���������d�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_5/MatMulMatMuldense_4/IdentityN:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_5/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_5/mulMuldense_5/beta:output:0dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_5/SigmoidSigmoiddense_5/mul:z:0*
T0*'
_output_shapes
:���������du
dense_5/mul_1Muldense_5/BiasAdd:output:0dense_5/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_5/IdentityIdentitydense_5/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_5/IdentityN	IdentityNdense_5/mul_1:z:0dense_5/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103395*:
_output_shapes(
&:���������d:���������d�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_6/MatMulMatMuldense_5/IdentityN:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_6/mulMuldense_6/beta:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_6/SigmoidSigmoiddense_6/mul:z:0*
T0*'
_output_shapes
:���������du
dense_6/mul_1Muldense_6/BiasAdd:output:0dense_6/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_6/IdentityIdentitydense_6/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_6/IdentityN	IdentityNdense_6/mul_1:z:0dense_6/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103409*:
_output_shapes(
&:���������d:���������d�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_7/MatMulMatMuldense_6/IdentityN:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_7/mulMuldense_7/beta:output:0dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_7/SigmoidSigmoiddense_7/mul:z:0*
T0*'
_output_shapes
:���������du
dense_7/mul_1Muldense_7/BiasAdd:output:0dense_7/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_7/IdentityIdentitydense_7/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_7/IdentityN	IdentityNdense_7/mul_1:z:0dense_7/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103423*:
_output_shapes(
&:���������d:���������d�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_8/MatMulMatMuldense_7/IdentityN:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_8/mulMuldense_8/beta:output:0dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_8/SigmoidSigmoiddense_8/mul:z:0*
T0*'
_output_shapes
:���������du
dense_8/mul_1Muldense_8/BiasAdd:output:0dense_8/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_8/IdentityIdentitydense_8/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_8/IdentityN	IdentityNdense_8/mul_1:z:0dense_8/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103437*:
_output_shapes(
&:���������d:���������d�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_9/MatMulMatMuldense_8/IdentityN:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_9/mulMuldense_9/beta:output:0dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_9/SigmoidSigmoiddense_9/mul:z:0*
T0*'
_output_shapes
:���������du
dense_9/mul_1Muldense_9/BiasAdd:output:0dense_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_9/IdentityIdentitydense_9/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_9/IdentityN	IdentityNdense_9/mul_1:z:0dense_9/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103451*:
_output_shapes(
&:���������d:���������d�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_10/MatMulMatMuldense_9/IdentityN:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_10/mulMuldense_10/beta:output:0dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_10/SigmoidSigmoiddense_10/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_10/mul_1Muldense_10/BiasAdd:output:0dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_10/IdentityIdentitydense_10/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_10/IdentityN	IdentityNdense_10/mul_1:z:0dense_10/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103465*:
_output_shapes(
&:���������d:���������d�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_11/MatMulMatMuldense_10/IdentityN:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_11/mulMuldense_11/beta:output:0dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_11/SigmoidSigmoiddense_11/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_11/mul_1Muldense_11/BiasAdd:output:0dense_11/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_11/IdentityIdentitydense_11/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_11/IdentityN	IdentityNdense_11/mul_1:z:0dense_11/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103479*:
_output_shapes(
&:���������d:���������d�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_12/MatMulMatMuldense_11/IdentityN:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_12/mulMuldense_12/beta:output:0dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_12/SigmoidSigmoiddense_12/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_12/mul_1Muldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_12/IdentityIdentitydense_12/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103493*:
_output_shapes(
&:���������d:���������d�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_13/MatMulMatMuldense_12/IdentityN:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_13/mulMuldense_13/beta:output:0dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_13/SigmoidSigmoiddense_13/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_13/mul_1Muldense_13/BiasAdd:output:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_13/IdentityIdentitydense_13/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103507*:
_output_shapes(
&:���������d:���������d�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_14/MatMulMatMuldense_13/IdentityN:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_14/mulMuldense_14/beta:output:0dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_14/SigmoidSigmoiddense_14/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_14/mul_1Muldense_14/BiasAdd:output:0dense_14/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_14/IdentityIdentitydense_14/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103521*:
_output_shapes(
&:���������d:���������d�
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_15/MatMulMatMuldense_14/IdentityN:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_15/mulMuldense_15/beta:output:0dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_15/SigmoidSigmoiddense_15/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_15/mul_1Muldense_15/BiasAdd:output:0dense_15/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_15/IdentityIdentitydense_15/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103535*:
_output_shapes(
&:���������d:���������d�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_16/MatMulMatMuldense_15/IdentityN:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_16/mulMuldense_16/beta:output:0dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_16/SigmoidSigmoiddense_16/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_16/mul_1Muldense_16/BiasAdd:output:0dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_16/IdentityIdentitydense_16/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103549*:
_output_shapes(
&:���������d:���������d�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_17/MatMulMatMuldense_16/IdentityN:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_17/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_17/mulMuldense_17/beta:output:0dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_17/SigmoidSigmoiddense_17/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_17/mul_1Muldense_17/BiasAdd:output:0dense_17/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_17/IdentityIdentitydense_17/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_17/IdentityN	IdentityNdense_17/mul_1:z:0dense_17/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103563*:
_output_shapes(
&:���������d:���������d�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_18/MatMulMatMuldense_17/IdentityN:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_18/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_18/mulMuldense_18/beta:output:0dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_18/SigmoidSigmoiddense_18/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_18/mul_1Muldense_18/BiasAdd:output:0dense_18/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_18/IdentityIdentitydense_18/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_18/IdentityN	IdentityNdense_18/mul_1:z:0dense_18/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103577*:
_output_shapes(
&:���������d:���������d�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_19/MatMulMatMuldense_18/IdentityN:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_19/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_19/mulMuldense_19/beta:output:0dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_19/SigmoidSigmoiddense_19/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_19/mul_1Muldense_19/BiasAdd:output:0dense_19/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_19/IdentityIdentitydense_19/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_19/IdentityN	IdentityNdense_19/mul_1:z:0dense_19/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103591*:
_output_shapes(
&:���������d:���������d�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0�
dense_20/MatMulMatMuldense_19/IdentityN:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������
i
IdentityIdentitydense_20/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
D__inference_dense_18_layer_call_and_return_conditional_losses_102090

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-102082*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_103910

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_101802

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101794*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106131
result_grads_0
result_grads_1
mul_dense_16_beta
mul_dense_16_biasadd
identityv
mulMulmul_dense_16_betamul_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_16_betamul_dense_16_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
)__inference_dense_19_layer_call_fn_104432

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_102114o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_9_layer_call_and_return_conditional_losses_104180

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104172*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_8_layer_call_and_return_conditional_losses_101850

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101842*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104799
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105411
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_19_layer_call_and_return_conditional_losses_104450

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104442*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105303
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_101682

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101674*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
F__inference_sequential_layer_call_and_return_conditional_losses_103899

inputs7
$dense_matmul_readvariableop_resource:	�d3
%dense_biasadd_readvariableop_resource:d8
&dense_1_matmul_readvariableop_resource:dd5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:dd5
'dense_2_biasadd_readvariableop_resource:d8
&dense_3_matmul_readvariableop_resource:dd5
'dense_3_biasadd_readvariableop_resource:d8
&dense_4_matmul_readvariableop_resource:dd5
'dense_4_biasadd_readvariableop_resource:d8
&dense_5_matmul_readvariableop_resource:dd5
'dense_5_biasadd_readvariableop_resource:d8
&dense_6_matmul_readvariableop_resource:dd5
'dense_6_biasadd_readvariableop_resource:d8
&dense_7_matmul_readvariableop_resource:dd5
'dense_7_biasadd_readvariableop_resource:d8
&dense_8_matmul_readvariableop_resource:dd5
'dense_8_biasadd_readvariableop_resource:d8
&dense_9_matmul_readvariableop_resource:dd5
'dense_9_biasadd_readvariableop_resource:d9
'dense_10_matmul_readvariableop_resource:dd6
(dense_10_biasadd_readvariableop_resource:d9
'dense_11_matmul_readvariableop_resource:dd6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:dd6
(dense_12_biasadd_readvariableop_resource:d9
'dense_13_matmul_readvariableop_resource:dd6
(dense_13_biasadd_readvariableop_resource:d9
'dense_14_matmul_readvariableop_resource:dd6
(dense_14_biasadd_readvariableop_resource:d9
'dense_15_matmul_readvariableop_resource:dd6
(dense_15_biasadd_readvariableop_resource:d9
'dense_16_matmul_readvariableop_resource:dd6
(dense_16_biasadd_readvariableop_resource:d9
'dense_17_matmul_readvariableop_resource:dd6
(dense_17_biasadd_readvariableop_resource:d9
'dense_18_matmul_readvariableop_resource:dd6
(dense_18_biasadd_readvariableop_resource:d9
'dense_19_matmul_readvariableop_resource:dd6
(dense_19_biasadd_readvariableop_resource:d9
'dense_20_matmul_readvariableop_resource:d
6
(dense_20_biasadd_readvariableop_resource:

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dO

dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	dense/mulMuldense/beta:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������dY
dense/SigmoidSigmoiddense/mul:z:0*
T0*'
_output_shapes
:���������do
dense/mul_1Muldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������d]
dense/IdentityIdentitydense/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense/IdentityN	IdentityNdense/mul_1:z:0dense/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103618*:
_output_shapes(
&:���������d:���������d�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:���������du
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103632*:
_output_shapes(
&:���������d:���������d�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_2/MatMulMatMuldense_1/IdentityN:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_2/mulMuldense_2/beta:output:0dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_2/SigmoidSigmoiddense_2/mul:z:0*
T0*'
_output_shapes
:���������du
dense_2/mul_1Muldense_2/BiasAdd:output:0dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_2/IdentityIdentitydense_2/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_2/IdentityN	IdentityNdense_2/mul_1:z:0dense_2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103646*:
_output_shapes(
&:���������d:���������d�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_3/MatMulMatMuldense_2/IdentityN:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_3/mulMuldense_3/beta:output:0dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_3/SigmoidSigmoiddense_3/mul:z:0*
T0*'
_output_shapes
:���������du
dense_3/mul_1Muldense_3/BiasAdd:output:0dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_3/IdentityIdentitydense_3/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_3/IdentityN	IdentityNdense_3/mul_1:z:0dense_3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103660*:
_output_shapes(
&:���������d:���������d�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_4/MatMulMatMuldense_3/IdentityN:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_4/mulMuldense_4/beta:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_4/SigmoidSigmoiddense_4/mul:z:0*
T0*'
_output_shapes
:���������du
dense_4/mul_1Muldense_4/BiasAdd:output:0dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_4/IdentityIdentitydense_4/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_4/IdentityN	IdentityNdense_4/mul_1:z:0dense_4/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103674*:
_output_shapes(
&:���������d:���������d�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_5/MatMulMatMuldense_4/IdentityN:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_5/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_5/mulMuldense_5/beta:output:0dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_5/SigmoidSigmoiddense_5/mul:z:0*
T0*'
_output_shapes
:���������du
dense_5/mul_1Muldense_5/BiasAdd:output:0dense_5/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_5/IdentityIdentitydense_5/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_5/IdentityN	IdentityNdense_5/mul_1:z:0dense_5/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103688*:
_output_shapes(
&:���������d:���������d�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_6/MatMulMatMuldense_5/IdentityN:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_6/mulMuldense_6/beta:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_6/SigmoidSigmoiddense_6/mul:z:0*
T0*'
_output_shapes
:���������du
dense_6/mul_1Muldense_6/BiasAdd:output:0dense_6/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_6/IdentityIdentitydense_6/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_6/IdentityN	IdentityNdense_6/mul_1:z:0dense_6/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103702*:
_output_shapes(
&:���������d:���������d�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_7/MatMulMatMuldense_6/IdentityN:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_7/mulMuldense_7/beta:output:0dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_7/SigmoidSigmoiddense_7/mul:z:0*
T0*'
_output_shapes
:���������du
dense_7/mul_1Muldense_7/BiasAdd:output:0dense_7/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_7/IdentityIdentitydense_7/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_7/IdentityN	IdentityNdense_7/mul_1:z:0dense_7/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103716*:
_output_shapes(
&:���������d:���������d�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_8/MatMulMatMuldense_7/IdentityN:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_8/mulMuldense_8/beta:output:0dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_8/SigmoidSigmoiddense_8/mul:z:0*
T0*'
_output_shapes
:���������du
dense_8/mul_1Muldense_8/BiasAdd:output:0dense_8/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_8/IdentityIdentitydense_8/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_8/IdentityN	IdentityNdense_8/mul_1:z:0dense_8/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103730*:
_output_shapes(
&:���������d:���������d�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_9/MatMulMatMuldense_8/IdentityN:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dQ
dense_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_9/mulMuldense_9/beta:output:0dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������d]
dense_9/SigmoidSigmoiddense_9/mul:z:0*
T0*'
_output_shapes
:���������du
dense_9/mul_1Muldense_9/BiasAdd:output:0dense_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������da
dense_9/IdentityIdentitydense_9/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_9/IdentityN	IdentityNdense_9/mul_1:z:0dense_9/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103744*:
_output_shapes(
&:���������d:���������d�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_10/MatMulMatMuldense_9/IdentityN:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_10/mulMuldense_10/beta:output:0dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_10/SigmoidSigmoiddense_10/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_10/mul_1Muldense_10/BiasAdd:output:0dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_10/IdentityIdentitydense_10/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_10/IdentityN	IdentityNdense_10/mul_1:z:0dense_10/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103758*:
_output_shapes(
&:���������d:���������d�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_11/MatMulMatMuldense_10/IdentityN:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_11/mulMuldense_11/beta:output:0dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_11/SigmoidSigmoiddense_11/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_11/mul_1Muldense_11/BiasAdd:output:0dense_11/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_11/IdentityIdentitydense_11/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_11/IdentityN	IdentityNdense_11/mul_1:z:0dense_11/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103772*:
_output_shapes(
&:���������d:���������d�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_12/MatMulMatMuldense_11/IdentityN:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_12/mulMuldense_12/beta:output:0dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_12/SigmoidSigmoiddense_12/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_12/mul_1Muldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_12/IdentityIdentitydense_12/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103786*:
_output_shapes(
&:���������d:���������d�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_13/MatMulMatMuldense_12/IdentityN:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_13/mulMuldense_13/beta:output:0dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_13/SigmoidSigmoiddense_13/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_13/mul_1Muldense_13/BiasAdd:output:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_13/IdentityIdentitydense_13/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103800*:
_output_shapes(
&:���������d:���������d�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_14/MatMulMatMuldense_13/IdentityN:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_14/mulMuldense_14/beta:output:0dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_14/SigmoidSigmoiddense_14/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_14/mul_1Muldense_14/BiasAdd:output:0dense_14/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_14/IdentityIdentitydense_14/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103814*:
_output_shapes(
&:���������d:���������d�
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_15/MatMulMatMuldense_14/IdentityN:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_15/mulMuldense_15/beta:output:0dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_15/SigmoidSigmoiddense_15/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_15/mul_1Muldense_15/BiasAdd:output:0dense_15/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_15/IdentityIdentitydense_15/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103828*:
_output_shapes(
&:���������d:���������d�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_16/MatMulMatMuldense_15/IdentityN:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_16/mulMuldense_16/beta:output:0dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_16/SigmoidSigmoiddense_16/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_16/mul_1Muldense_16/BiasAdd:output:0dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_16/IdentityIdentitydense_16/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103842*:
_output_shapes(
&:���������d:���������d�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_17/MatMulMatMuldense_16/IdentityN:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_17/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_17/mulMuldense_17/beta:output:0dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_17/SigmoidSigmoiddense_17/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_17/mul_1Muldense_17/BiasAdd:output:0dense_17/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_17/IdentityIdentitydense_17/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_17/IdentityN	IdentityNdense_17/mul_1:z:0dense_17/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103856*:
_output_shapes(
&:���������d:���������d�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_18/MatMulMatMuldense_17/IdentityN:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_18/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_18/mulMuldense_18/beta:output:0dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_18/SigmoidSigmoiddense_18/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_18/mul_1Muldense_18/BiasAdd:output:0dense_18/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_18/IdentityIdentitydense_18/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_18/IdentityN	IdentityNdense_18/mul_1:z:0dense_18/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103870*:
_output_shapes(
&:���������d:���������d�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_19/MatMulMatMuldense_18/IdentityN:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dR
dense_19/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
dense_19/mulMuldense_19/beta:output:0dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������d_
dense_19/SigmoidSigmoiddense_19/mul:z:0*
T0*'
_output_shapes
:���������dx
dense_19/mul_1Muldense_19/BiasAdd:output:0dense_19/Sigmoid:y:0*
T0*'
_output_shapes
:���������dc
dense_19/IdentityIdentitydense_19/mul_1:z:0*
T0*'
_output_shapes
:���������d�
dense_19/IdentityN	IdentityNdense_19/mul_1:z:0dense_19/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103884*:
_output_shapes(
&:���������d:���������d�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0�
dense_20/MatMulMatMuldense_19/IdentityN:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_20/SoftmaxSoftmaxdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������
i
IdentityIdentitydense_20/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105069
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_13_layer_call_and_return_conditional_losses_104288

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104280*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105897
result_grads_0
result_grads_1
mul_dense_3_beta
mul_dense_3_biasadd
identityt
mulMulmul_dense_3_betamul_dense_3_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_3_betamul_dense_3_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105771
result_grads_0
result_grads_1
mul_dense_16_beta
mul_dense_16_biasadd
identityv
mulMulmul_dense_16_betamul_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_16_betamul_dense_16_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_16_layer_call_and_return_conditional_losses_102042

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-102034*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105105
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106419
result_grads_0
result_grads_1 
mul_sequential_dense_12_beta#
mul_sequential_dense_12_biasadd
identity�
mulMulmul_sequential_dense_12_betamul_sequential_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_12_betamul_sequential_dense_12_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106023
result_grads_0
result_grads_1
mul_dense_10_beta
mul_dense_10_biasadd
identityv
mulMulmul_dense_10_betamul_dense_10_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_10_betamul_dense_10_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105753
result_grads_0
result_grads_1
mul_dense_15_beta
mul_dense_15_biasadd
identityv
mulMulmul_dense_15_betamul_dense_15_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_15_betamul_dense_15_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106059
result_grads_0
result_grads_1
mul_dense_12_beta
mul_dense_12_biasadd
identityv
mulMulmul_dense_12_betamul_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_12_betamul_dense_12_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_104045

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104037*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106455
result_grads_0
result_grads_1 
mul_sequential_dense_14_beta#
mul_sequential_dense_14_biasadd
identity�
mulMulmul_sequential_dense_14_betamul_sequential_dense_14_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_14_betamul_sequential_dense_14_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_104853
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
&__inference_dense_layer_call_fn_103919

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101658o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_dense_14_layer_call_and_return_conditional_losses_104315

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104307*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_11_layer_call_fn_104216

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_101922o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_101754

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101746*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_106203
result_grads_0
result_grads_1
mul_sequential_dense_beta 
mul_sequential_dense_biasadd
identity�
mulMulmul_sequential_dense_betamul_sequential_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dw
mul_1Mulmul_sequential_dense_betamul_sequential_dense_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105681
result_grads_0
result_grads_1
mul_dense_11_beta
mul_dense_11_biasadd
identityv
mulMulmul_dense_11_betamul_dense_11_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_11_betamul_dense_11_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
��
�7
__inference__traced_save_106701
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop5
1savev2_nadam_dense_1_kernel_m_read_readvariableop3
/savev2_nadam_dense_1_bias_m_read_readvariableop5
1savev2_nadam_dense_2_kernel_m_read_readvariableop3
/savev2_nadam_dense_2_bias_m_read_readvariableop5
1savev2_nadam_dense_3_kernel_m_read_readvariableop3
/savev2_nadam_dense_3_bias_m_read_readvariableop5
1savev2_nadam_dense_4_kernel_m_read_readvariableop3
/savev2_nadam_dense_4_bias_m_read_readvariableop5
1savev2_nadam_dense_5_kernel_m_read_readvariableop3
/savev2_nadam_dense_5_bias_m_read_readvariableop5
1savev2_nadam_dense_6_kernel_m_read_readvariableop3
/savev2_nadam_dense_6_bias_m_read_readvariableop5
1savev2_nadam_dense_7_kernel_m_read_readvariableop3
/savev2_nadam_dense_7_bias_m_read_readvariableop5
1savev2_nadam_dense_8_kernel_m_read_readvariableop3
/savev2_nadam_dense_8_bias_m_read_readvariableop5
1savev2_nadam_dense_9_kernel_m_read_readvariableop3
/savev2_nadam_dense_9_bias_m_read_readvariableop6
2savev2_nadam_dense_10_kernel_m_read_readvariableop4
0savev2_nadam_dense_10_bias_m_read_readvariableop6
2savev2_nadam_dense_11_kernel_m_read_readvariableop4
0savev2_nadam_dense_11_bias_m_read_readvariableop6
2savev2_nadam_dense_12_kernel_m_read_readvariableop4
0savev2_nadam_dense_12_bias_m_read_readvariableop6
2savev2_nadam_dense_13_kernel_m_read_readvariableop4
0savev2_nadam_dense_13_bias_m_read_readvariableop6
2savev2_nadam_dense_14_kernel_m_read_readvariableop4
0savev2_nadam_dense_14_bias_m_read_readvariableop6
2savev2_nadam_dense_15_kernel_m_read_readvariableop4
0savev2_nadam_dense_15_bias_m_read_readvariableop6
2savev2_nadam_dense_16_kernel_m_read_readvariableop4
0savev2_nadam_dense_16_bias_m_read_readvariableop6
2savev2_nadam_dense_17_kernel_m_read_readvariableop4
0savev2_nadam_dense_17_bias_m_read_readvariableop6
2savev2_nadam_dense_18_kernel_m_read_readvariableop4
0savev2_nadam_dense_18_bias_m_read_readvariableop6
2savev2_nadam_dense_19_kernel_m_read_readvariableop4
0savev2_nadam_dense_19_bias_m_read_readvariableop6
2savev2_nadam_dense_20_kernel_m_read_readvariableop4
0savev2_nadam_dense_20_bias_m_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop5
1savev2_nadam_dense_1_kernel_v_read_readvariableop3
/savev2_nadam_dense_1_bias_v_read_readvariableop5
1savev2_nadam_dense_2_kernel_v_read_readvariableop3
/savev2_nadam_dense_2_bias_v_read_readvariableop5
1savev2_nadam_dense_3_kernel_v_read_readvariableop3
/savev2_nadam_dense_3_bias_v_read_readvariableop5
1savev2_nadam_dense_4_kernel_v_read_readvariableop3
/savev2_nadam_dense_4_bias_v_read_readvariableop5
1savev2_nadam_dense_5_kernel_v_read_readvariableop3
/savev2_nadam_dense_5_bias_v_read_readvariableop5
1savev2_nadam_dense_6_kernel_v_read_readvariableop3
/savev2_nadam_dense_6_bias_v_read_readvariableop5
1savev2_nadam_dense_7_kernel_v_read_readvariableop3
/savev2_nadam_dense_7_bias_v_read_readvariableop5
1savev2_nadam_dense_8_kernel_v_read_readvariableop3
/savev2_nadam_dense_8_bias_v_read_readvariableop5
1savev2_nadam_dense_9_kernel_v_read_readvariableop3
/savev2_nadam_dense_9_bias_v_read_readvariableop6
2savev2_nadam_dense_10_kernel_v_read_readvariableop4
0savev2_nadam_dense_10_bias_v_read_readvariableop6
2savev2_nadam_dense_11_kernel_v_read_readvariableop4
0savev2_nadam_dense_11_bias_v_read_readvariableop6
2savev2_nadam_dense_12_kernel_v_read_readvariableop4
0savev2_nadam_dense_12_bias_v_read_readvariableop6
2savev2_nadam_dense_13_kernel_v_read_readvariableop4
0savev2_nadam_dense_13_bias_v_read_readvariableop6
2savev2_nadam_dense_14_kernel_v_read_readvariableop4
0savev2_nadam_dense_14_bias_v_read_readvariableop6
2savev2_nadam_dense_15_kernel_v_read_readvariableop4
0savev2_nadam_dense_15_bias_v_read_readvariableop6
2savev2_nadam_dense_16_kernel_v_read_readvariableop4
0savev2_nadam_dense_16_bias_v_read_readvariableop6
2savev2_nadam_dense_17_kernel_v_read_readvariableop4
0savev2_nadam_dense_17_bias_v_read_readvariableop6
2savev2_nadam_dense_18_kernel_v_read_readvariableop4
0savev2_nadam_dense_18_bias_v_read_readvariableop6
2savev2_nadam_dense_19_kernel_v_read_readvariableop4
0savev2_nadam_dense_19_bias_v_read_readvariableop6
2savev2_nadam_dense_20_kernel_v_read_readvariableop4
0savev2_nadam_dense_20_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �N
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�M
value�MB�M�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop1savev2_nadam_dense_1_kernel_m_read_readvariableop/savev2_nadam_dense_1_bias_m_read_readvariableop1savev2_nadam_dense_2_kernel_m_read_readvariableop/savev2_nadam_dense_2_bias_m_read_readvariableop1savev2_nadam_dense_3_kernel_m_read_readvariableop/savev2_nadam_dense_3_bias_m_read_readvariableop1savev2_nadam_dense_4_kernel_m_read_readvariableop/savev2_nadam_dense_4_bias_m_read_readvariableop1savev2_nadam_dense_5_kernel_m_read_readvariableop/savev2_nadam_dense_5_bias_m_read_readvariableop1savev2_nadam_dense_6_kernel_m_read_readvariableop/savev2_nadam_dense_6_bias_m_read_readvariableop1savev2_nadam_dense_7_kernel_m_read_readvariableop/savev2_nadam_dense_7_bias_m_read_readvariableop1savev2_nadam_dense_8_kernel_m_read_readvariableop/savev2_nadam_dense_8_bias_m_read_readvariableop1savev2_nadam_dense_9_kernel_m_read_readvariableop/savev2_nadam_dense_9_bias_m_read_readvariableop2savev2_nadam_dense_10_kernel_m_read_readvariableop0savev2_nadam_dense_10_bias_m_read_readvariableop2savev2_nadam_dense_11_kernel_m_read_readvariableop0savev2_nadam_dense_11_bias_m_read_readvariableop2savev2_nadam_dense_12_kernel_m_read_readvariableop0savev2_nadam_dense_12_bias_m_read_readvariableop2savev2_nadam_dense_13_kernel_m_read_readvariableop0savev2_nadam_dense_13_bias_m_read_readvariableop2savev2_nadam_dense_14_kernel_m_read_readvariableop0savev2_nadam_dense_14_bias_m_read_readvariableop2savev2_nadam_dense_15_kernel_m_read_readvariableop0savev2_nadam_dense_15_bias_m_read_readvariableop2savev2_nadam_dense_16_kernel_m_read_readvariableop0savev2_nadam_dense_16_bias_m_read_readvariableop2savev2_nadam_dense_17_kernel_m_read_readvariableop0savev2_nadam_dense_17_bias_m_read_readvariableop2savev2_nadam_dense_18_kernel_m_read_readvariableop0savev2_nadam_dense_18_bias_m_read_readvariableop2savev2_nadam_dense_19_kernel_m_read_readvariableop0savev2_nadam_dense_19_bias_m_read_readvariableop2savev2_nadam_dense_20_kernel_m_read_readvariableop0savev2_nadam_dense_20_bias_m_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop1savev2_nadam_dense_1_kernel_v_read_readvariableop/savev2_nadam_dense_1_bias_v_read_readvariableop1savev2_nadam_dense_2_kernel_v_read_readvariableop/savev2_nadam_dense_2_bias_v_read_readvariableop1savev2_nadam_dense_3_kernel_v_read_readvariableop/savev2_nadam_dense_3_bias_v_read_readvariableop1savev2_nadam_dense_4_kernel_v_read_readvariableop/savev2_nadam_dense_4_bias_v_read_readvariableop1savev2_nadam_dense_5_kernel_v_read_readvariableop/savev2_nadam_dense_5_bias_v_read_readvariableop1savev2_nadam_dense_6_kernel_v_read_readvariableop/savev2_nadam_dense_6_bias_v_read_readvariableop1savev2_nadam_dense_7_kernel_v_read_readvariableop/savev2_nadam_dense_7_bias_v_read_readvariableop1savev2_nadam_dense_8_kernel_v_read_readvariableop/savev2_nadam_dense_8_bias_v_read_readvariableop1savev2_nadam_dense_9_kernel_v_read_readvariableop/savev2_nadam_dense_9_bias_v_read_readvariableop2savev2_nadam_dense_10_kernel_v_read_readvariableop0savev2_nadam_dense_10_bias_v_read_readvariableop2savev2_nadam_dense_11_kernel_v_read_readvariableop0savev2_nadam_dense_11_bias_v_read_readvariableop2savev2_nadam_dense_12_kernel_v_read_readvariableop0savev2_nadam_dense_12_bias_v_read_readvariableop2savev2_nadam_dense_13_kernel_v_read_readvariableop0savev2_nadam_dense_13_bias_v_read_readvariableop2savev2_nadam_dense_14_kernel_v_read_readvariableop0savev2_nadam_dense_14_bias_v_read_readvariableop2savev2_nadam_dense_15_kernel_v_read_readvariableop0savev2_nadam_dense_15_bias_v_read_readvariableop2savev2_nadam_dense_16_kernel_v_read_readvariableop0savev2_nadam_dense_16_bias_v_read_readvariableop2savev2_nadam_dense_17_kernel_v_read_readvariableop0savev2_nadam_dense_17_bias_v_read_readvariableop2savev2_nadam_dense_18_kernel_v_read_readvariableop0savev2_nadam_dense_18_bias_v_read_readvariableop2savev2_nadam_dense_19_kernel_v_read_readvariableop0savev2_nadam_dense_19_bias_v_read_readvariableop2savev2_nadam_dense_20_kernel_v_read_readvariableop0savev2_nadam_dense_20_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d
:
: : : : : : : : : : :	�d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d
:
:	�d:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:dd:d:d
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$	 

_output_shapes

:dd: 


_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd:  

_output_shapes
:d:$! 

_output_shapes

:dd: "

_output_shapes
:d:$# 

_output_shapes

:dd: $

_output_shapes
:d:$% 

_output_shapes

:dd: &

_output_shapes
:d:$' 

_output_shapes

:dd: (

_output_shapes
:d:$) 

_output_shapes

:d
: *

_output_shapes
:
:+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :%5!

_output_shapes
:	�d: 6

_output_shapes
:d:$7 

_output_shapes

:dd: 8

_output_shapes
:d:$9 

_output_shapes

:dd: :

_output_shapes
:d:$; 

_output_shapes

:dd: <

_output_shapes
:d:$= 

_output_shapes

:dd: >

_output_shapes
:d:$? 

_output_shapes

:dd: @

_output_shapes
:d:$A 

_output_shapes

:dd: B

_output_shapes
:d:$C 

_output_shapes

:dd: D

_output_shapes
:d:$E 

_output_shapes

:dd: F

_output_shapes
:d:$G 

_output_shapes

:dd: H

_output_shapes
:d:$I 

_output_shapes

:dd: J

_output_shapes
:d:$K 

_output_shapes

:dd: L

_output_shapes
:d:$M 

_output_shapes

:dd: N

_output_shapes
:d:$O 

_output_shapes

:dd: P

_output_shapes
:d:$Q 

_output_shapes

:dd: R

_output_shapes
:d:$S 

_output_shapes

:dd: T

_output_shapes
:d:$U 

_output_shapes

:dd: V

_output_shapes
:d:$W 

_output_shapes

:dd: X

_output_shapes
:d:$Y 

_output_shapes

:dd: Z

_output_shapes
:d:$[ 

_output_shapes

:dd: \

_output_shapes
:d:$] 

_output_shapes

:d
: ^

_output_shapes
:
:%_!

_output_shapes
:	�d: `

_output_shapes
:d:$a 

_output_shapes

:dd: b

_output_shapes
:d:$c 

_output_shapes

:dd: d

_output_shapes
:d:$e 

_output_shapes

:dd: f

_output_shapes
:d:$g 

_output_shapes

:dd: h

_output_shapes
:d:$i 

_output_shapes

:dd: j

_output_shapes
:d:$k 

_output_shapes

:dd: l

_output_shapes
:d:$m 

_output_shapes

:dd: n

_output_shapes
:d:$o 

_output_shapes

:dd: p

_output_shapes
:d:$q 

_output_shapes

:dd: r

_output_shapes
:d:$s 

_output_shapes

:dd: t

_output_shapes
:d:$u 

_output_shapes

:dd: v

_output_shapes
:d:$w 

_output_shapes

:dd: x

_output_shapes
:d:$y 

_output_shapes

:dd: z

_output_shapes
:d:${ 

_output_shapes

:dd: |

_output_shapes
:d:$} 

_output_shapes

:dd: ~

_output_shapes
:d:$ 

_output_shapes

:dd:!�

_output_shapes
:d:%� 

_output_shapes

:dd:!�

_output_shapes
:d:%� 

_output_shapes

:dd:!�

_output_shapes
:d:%� 

_output_shapes

:dd:!�

_output_shapes
:d:%� 

_output_shapes

:d
:!�

_output_shapes
:
:�

_output_shapes
: 
�
�
#__inference_internal_grad_fn_106113
result_grads_0
result_grads_1
mul_dense_15_beta
mul_dense_15_biasadd
identityv
mulMulmul_dense_15_betamul_dense_15_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_15_betamul_dense_15_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105645
result_grads_0
result_grads_1
mul_dense_9_beta
mul_dense_9_biasadd
identityt
mulMulmul_dense_9_betamul_dense_9_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_9_betamul_dense_9_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
)__inference_dense_20_layer_call_fn_104459

inputs
unknown:d

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_102131o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_15_layer_call_fn_104324

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_102018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104997
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_103964

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103956*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105717
result_grads_0
result_grads_1
mul_dense_13_beta
mul_dense_13_biasadd
identityv
mulMulmul_dense_13_betamul_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dg
mul_1Mulmul_dense_13_betamul_dense_13_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
D__inference_dense_19_layer_call_and_return_conditional_losses_102114

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-102106*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
(__inference_dense_7_layer_call_fn_104108

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_101826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104925
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
)__inference_dense_14_layer_call_fn_104297

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_101994o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105267
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
��
�S
"__inference__traced_restore_107119
file_prefix0
assignvariableop_dense_kernel:	�d+
assignvariableop_1_dense_bias:d3
!assignvariableop_2_dense_1_kernel:dd-
assignvariableop_3_dense_1_bias:d3
!assignvariableop_4_dense_2_kernel:dd-
assignvariableop_5_dense_2_bias:d3
!assignvariableop_6_dense_3_kernel:dd-
assignvariableop_7_dense_3_bias:d3
!assignvariableop_8_dense_4_kernel:dd-
assignvariableop_9_dense_4_bias:d4
"assignvariableop_10_dense_5_kernel:dd.
 assignvariableop_11_dense_5_bias:d4
"assignvariableop_12_dense_6_kernel:dd.
 assignvariableop_13_dense_6_bias:d4
"assignvariableop_14_dense_7_kernel:dd.
 assignvariableop_15_dense_7_bias:d4
"assignvariableop_16_dense_8_kernel:dd.
 assignvariableop_17_dense_8_bias:d4
"assignvariableop_18_dense_9_kernel:dd.
 assignvariableop_19_dense_9_bias:d5
#assignvariableop_20_dense_10_kernel:dd/
!assignvariableop_21_dense_10_bias:d5
#assignvariableop_22_dense_11_kernel:dd/
!assignvariableop_23_dense_11_bias:d5
#assignvariableop_24_dense_12_kernel:dd/
!assignvariableop_25_dense_12_bias:d5
#assignvariableop_26_dense_13_kernel:dd/
!assignvariableop_27_dense_13_bias:d5
#assignvariableop_28_dense_14_kernel:dd/
!assignvariableop_29_dense_14_bias:d5
#assignvariableop_30_dense_15_kernel:dd/
!assignvariableop_31_dense_15_bias:d5
#assignvariableop_32_dense_16_kernel:dd/
!assignvariableop_33_dense_16_bias:d5
#assignvariableop_34_dense_17_kernel:dd/
!assignvariableop_35_dense_17_bias:d5
#assignvariableop_36_dense_18_kernel:dd/
!assignvariableop_37_dense_18_bias:d5
#assignvariableop_38_dense_19_kernel:dd/
!assignvariableop_39_dense_19_bias:d5
#assignvariableop_40_dense_20_kernel:d
/
!assignvariableop_41_dense_20_bias:
(
assignvariableop_42_nadam_iter:	 *
 assignvariableop_43_nadam_beta_1: *
 assignvariableop_44_nadam_beta_2: )
assignvariableop_45_nadam_decay: 1
'assignvariableop_46_nadam_learning_rate: 2
(assignvariableop_47_nadam_momentum_cache: %
assignvariableop_48_total_1: %
assignvariableop_49_count_1: #
assignvariableop_50_total: #
assignvariableop_51_count: ;
(assignvariableop_52_nadam_dense_kernel_m:	�d4
&assignvariableop_53_nadam_dense_bias_m:d<
*assignvariableop_54_nadam_dense_1_kernel_m:dd6
(assignvariableop_55_nadam_dense_1_bias_m:d<
*assignvariableop_56_nadam_dense_2_kernel_m:dd6
(assignvariableop_57_nadam_dense_2_bias_m:d<
*assignvariableop_58_nadam_dense_3_kernel_m:dd6
(assignvariableop_59_nadam_dense_3_bias_m:d<
*assignvariableop_60_nadam_dense_4_kernel_m:dd6
(assignvariableop_61_nadam_dense_4_bias_m:d<
*assignvariableop_62_nadam_dense_5_kernel_m:dd6
(assignvariableop_63_nadam_dense_5_bias_m:d<
*assignvariableop_64_nadam_dense_6_kernel_m:dd6
(assignvariableop_65_nadam_dense_6_bias_m:d<
*assignvariableop_66_nadam_dense_7_kernel_m:dd6
(assignvariableop_67_nadam_dense_7_bias_m:d<
*assignvariableop_68_nadam_dense_8_kernel_m:dd6
(assignvariableop_69_nadam_dense_8_bias_m:d<
*assignvariableop_70_nadam_dense_9_kernel_m:dd6
(assignvariableop_71_nadam_dense_9_bias_m:d=
+assignvariableop_72_nadam_dense_10_kernel_m:dd7
)assignvariableop_73_nadam_dense_10_bias_m:d=
+assignvariableop_74_nadam_dense_11_kernel_m:dd7
)assignvariableop_75_nadam_dense_11_bias_m:d=
+assignvariableop_76_nadam_dense_12_kernel_m:dd7
)assignvariableop_77_nadam_dense_12_bias_m:d=
+assignvariableop_78_nadam_dense_13_kernel_m:dd7
)assignvariableop_79_nadam_dense_13_bias_m:d=
+assignvariableop_80_nadam_dense_14_kernel_m:dd7
)assignvariableop_81_nadam_dense_14_bias_m:d=
+assignvariableop_82_nadam_dense_15_kernel_m:dd7
)assignvariableop_83_nadam_dense_15_bias_m:d=
+assignvariableop_84_nadam_dense_16_kernel_m:dd7
)assignvariableop_85_nadam_dense_16_bias_m:d=
+assignvariableop_86_nadam_dense_17_kernel_m:dd7
)assignvariableop_87_nadam_dense_17_bias_m:d=
+assignvariableop_88_nadam_dense_18_kernel_m:dd7
)assignvariableop_89_nadam_dense_18_bias_m:d=
+assignvariableop_90_nadam_dense_19_kernel_m:dd7
)assignvariableop_91_nadam_dense_19_bias_m:d=
+assignvariableop_92_nadam_dense_20_kernel_m:d
7
)assignvariableop_93_nadam_dense_20_bias_m:
;
(assignvariableop_94_nadam_dense_kernel_v:	�d4
&assignvariableop_95_nadam_dense_bias_v:d<
*assignvariableop_96_nadam_dense_1_kernel_v:dd6
(assignvariableop_97_nadam_dense_1_bias_v:d<
*assignvariableop_98_nadam_dense_2_kernel_v:dd6
(assignvariableop_99_nadam_dense_2_bias_v:d=
+assignvariableop_100_nadam_dense_3_kernel_v:dd7
)assignvariableop_101_nadam_dense_3_bias_v:d=
+assignvariableop_102_nadam_dense_4_kernel_v:dd7
)assignvariableop_103_nadam_dense_4_bias_v:d=
+assignvariableop_104_nadam_dense_5_kernel_v:dd7
)assignvariableop_105_nadam_dense_5_bias_v:d=
+assignvariableop_106_nadam_dense_6_kernel_v:dd7
)assignvariableop_107_nadam_dense_6_bias_v:d=
+assignvariableop_108_nadam_dense_7_kernel_v:dd7
)assignvariableop_109_nadam_dense_7_bias_v:d=
+assignvariableop_110_nadam_dense_8_kernel_v:dd7
)assignvariableop_111_nadam_dense_8_bias_v:d=
+assignvariableop_112_nadam_dense_9_kernel_v:dd7
)assignvariableop_113_nadam_dense_9_bias_v:d>
,assignvariableop_114_nadam_dense_10_kernel_v:dd8
*assignvariableop_115_nadam_dense_10_bias_v:d>
,assignvariableop_116_nadam_dense_11_kernel_v:dd8
*assignvariableop_117_nadam_dense_11_bias_v:d>
,assignvariableop_118_nadam_dense_12_kernel_v:dd8
*assignvariableop_119_nadam_dense_12_bias_v:d>
,assignvariableop_120_nadam_dense_13_kernel_v:dd8
*assignvariableop_121_nadam_dense_13_bias_v:d>
,assignvariableop_122_nadam_dense_14_kernel_v:dd8
*assignvariableop_123_nadam_dense_14_bias_v:d>
,assignvariableop_124_nadam_dense_15_kernel_v:dd8
*assignvariableop_125_nadam_dense_15_bias_v:d>
,assignvariableop_126_nadam_dense_16_kernel_v:dd8
*assignvariableop_127_nadam_dense_16_bias_v:d>
,assignvariableop_128_nadam_dense_17_kernel_v:dd8
*assignvariableop_129_nadam_dense_17_bias_v:d>
,assignvariableop_130_nadam_dense_18_kernel_v:dd8
*assignvariableop_131_nadam_dense_18_bias_v:d>
,assignvariableop_132_nadam_dense_19_kernel_v:dd8
*assignvariableop_133_nadam_dense_19_bias_v:d>
,assignvariableop_134_nadam_dense_20_kernel_v:d
8
*assignvariableop_135_nadam_dense_20_bias_v:

identity_137��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�N
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�M
value�MB�M�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_12_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_12_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_13_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_13_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_17_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_17_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_18_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_18_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_19_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_19_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_20_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_dense_20_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_nadam_iterIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp assignvariableop_43_nadam_beta_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp assignvariableop_44_nadam_beta_2Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_nadam_decayIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_nadam_learning_rateIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_nadam_momentum_cacheIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_total_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_count_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_nadam_dense_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp&assignvariableop_53_nadam_dense_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp*assignvariableop_54_nadam_dense_1_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_nadam_dense_1_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp*assignvariableop_56_nadam_dense_2_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp(assignvariableop_57_nadam_dense_2_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_nadam_dense_3_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp(assignvariableop_59_nadam_dense_3_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp*assignvariableop_60_nadam_dense_4_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp(assignvariableop_61_nadam_dense_4_bias_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp*assignvariableop_62_nadam_dense_5_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp(assignvariableop_63_nadam_dense_5_bias_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp*assignvariableop_64_nadam_dense_6_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp(assignvariableop_65_nadam_dense_6_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp*assignvariableop_66_nadam_dense_7_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp(assignvariableop_67_nadam_dense_7_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp*assignvariableop_68_nadam_dense_8_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp(assignvariableop_69_nadam_dense_8_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp*assignvariableop_70_nadam_dense_9_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp(assignvariableop_71_nadam_dense_9_bias_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp+assignvariableop_72_nadam_dense_10_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp)assignvariableop_73_nadam_dense_10_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp+assignvariableop_74_nadam_dense_11_kernel_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp)assignvariableop_75_nadam_dense_11_bias_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp+assignvariableop_76_nadam_dense_12_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp)assignvariableop_77_nadam_dense_12_bias_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp+assignvariableop_78_nadam_dense_13_kernel_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp)assignvariableop_79_nadam_dense_13_bias_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp+assignvariableop_80_nadam_dense_14_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp)assignvariableop_81_nadam_dense_14_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp+assignvariableop_82_nadam_dense_15_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp)assignvariableop_83_nadam_dense_15_bias_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp+assignvariableop_84_nadam_dense_16_kernel_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp)assignvariableop_85_nadam_dense_16_bias_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp+assignvariableop_86_nadam_dense_17_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp)assignvariableop_87_nadam_dense_17_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp+assignvariableop_88_nadam_dense_18_kernel_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp)assignvariableop_89_nadam_dense_18_bias_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp+assignvariableop_90_nadam_dense_19_kernel_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp)assignvariableop_91_nadam_dense_19_bias_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp+assignvariableop_92_nadam_dense_20_kernel_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp)assignvariableop_93_nadam_dense_20_bias_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_nadam_dense_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp&assignvariableop_95_nadam_dense_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp*assignvariableop_96_nadam_dense_1_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp(assignvariableop_97_nadam_dense_1_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp*assignvariableop_98_nadam_dense_2_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp(assignvariableop_99_nadam_dense_2_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp+assignvariableop_100_nadam_dense_3_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp)assignvariableop_101_nadam_dense_3_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp+assignvariableop_102_nadam_dense_4_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp)assignvariableop_103_nadam_dense_4_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp+assignvariableop_104_nadam_dense_5_kernel_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp)assignvariableop_105_nadam_dense_5_bias_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp+assignvariableop_106_nadam_dense_6_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp)assignvariableop_107_nadam_dense_6_bias_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp+assignvariableop_108_nadam_dense_7_kernel_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp)assignvariableop_109_nadam_dense_7_bias_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp+assignvariableop_110_nadam_dense_8_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp)assignvariableop_111_nadam_dense_8_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp+assignvariableop_112_nadam_dense_9_kernel_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp)assignvariableop_113_nadam_dense_9_bias_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp,assignvariableop_114_nadam_dense_10_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp*assignvariableop_115_nadam_dense_10_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp,assignvariableop_116_nadam_dense_11_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp*assignvariableop_117_nadam_dense_11_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp,assignvariableop_118_nadam_dense_12_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp*assignvariableop_119_nadam_dense_12_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp,assignvariableop_120_nadam_dense_13_kernel_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp*assignvariableop_121_nadam_dense_13_bias_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp,assignvariableop_122_nadam_dense_14_kernel_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp*assignvariableop_123_nadam_dense_14_bias_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp,assignvariableop_124_nadam_dense_15_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp*assignvariableop_125_nadam_dense_15_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp,assignvariableop_126_nadam_dense_16_kernel_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp*assignvariableop_127_nadam_dense_16_bias_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp,assignvariableop_128_nadam_dense_17_kernel_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp*assignvariableop_129_nadam_dense_17_bias_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp,assignvariableop_130_nadam_dense_18_kernel_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp*assignvariableop_131_nadam_dense_18_bias_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp,assignvariableop_132_nadam_dense_19_kernel_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp*assignvariableop_133_nadam_dense_19_bias_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOp,assignvariableop_134_nadam_dense_20_kernel_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOp*assignvariableop_135_nadam_dense_20_bias_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_136Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_137IdentityIdentity_136:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_137Identity_137:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�j
�
F__inference_sequential_layer_call_and_return_conditional_losses_102642

inputs
dense_102536:	�d
dense_102538:d 
dense_1_102541:dd
dense_1_102543:d 
dense_2_102546:dd
dense_2_102548:d 
dense_3_102551:dd
dense_3_102553:d 
dense_4_102556:dd
dense_4_102558:d 
dense_5_102561:dd
dense_5_102563:d 
dense_6_102566:dd
dense_6_102568:d 
dense_7_102571:dd
dense_7_102573:d 
dense_8_102576:dd
dense_8_102578:d 
dense_9_102581:dd
dense_9_102583:d!
dense_10_102586:dd
dense_10_102588:d!
dense_11_102591:dd
dense_11_102593:d!
dense_12_102596:dd
dense_12_102598:d!
dense_13_102601:dd
dense_13_102603:d!
dense_14_102606:dd
dense_14_102608:d!
dense_15_102611:dd
dense_15_102613:d!
dense_16_102616:dd
dense_16_102618:d!
dense_17_102621:dd
dense_17_102623:d!
dense_18_102626:dd
dense_18_102628:d!
dense_19_102631:dd
dense_19_102633:d!
dense_20_102636:d

dense_20_102638:

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�dense_2/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101638�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_102536dense_102538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101658�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_102541dense_1_102543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101682�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_102546dense_2_102548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101706�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_102551dense_3_102553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_101730�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_102556dense_4_102558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_101754�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_102561dense_5_102563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_101778�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_102566dense_6_102568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_101802�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_102571dense_7_102573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_101826�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_102576dense_8_102578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_101850�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_102581dense_9_102583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_101874�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_102586dense_10_102588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_101898�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_102591dense_11_102593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_101922�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_102596dense_12_102598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_101946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_102601dense_13_102603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_101970�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_102606dense_14_102608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_101994�
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_102611dense_15_102613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_102018�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_102616dense_16_102618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_102042�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_102621dense_17_102623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_102066�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_102626dense_18_102628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_102090�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_102631dense_19_102633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_102114�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_102636dense_20_102638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_102131x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
D__inference_dense_11_layer_call_and_return_conditional_losses_104234

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104226*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_103991

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-103983*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
D__inference_dense_18_layer_call_and_return_conditional_losses_104423

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104415*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_105339
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_104099

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������d�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-104091*:
_output_shapes(
&:���������d:���������dc

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104907
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_105573
result_grads_0
result_grads_1
mul_dense_5_beta
mul_dense_5_biasadd
identityt
mulMulmul_dense_5_betamul_dense_5_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_5_betamul_dense_5_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
z
#__inference_internal_grad_fn_105141
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�j
�
F__inference_sequential_layer_call_and_return_conditional_losses_102138

inputs
dense_101659:	�d
dense_101661:d 
dense_1_101683:dd
dense_1_101685:d 
dense_2_101707:dd
dense_2_101709:d 
dense_3_101731:dd
dense_3_101733:d 
dense_4_101755:dd
dense_4_101757:d 
dense_5_101779:dd
dense_5_101781:d 
dense_6_101803:dd
dense_6_101805:d 
dense_7_101827:dd
dense_7_101829:d 
dense_8_101851:dd
dense_8_101853:d 
dense_9_101875:dd
dense_9_101877:d!
dense_10_101899:dd
dense_10_101901:d!
dense_11_101923:dd
dense_11_101925:d!
dense_12_101947:dd
dense_12_101949:d!
dense_13_101971:dd
dense_13_101973:d!
dense_14_101995:dd
dense_14_101997:d!
dense_15_102019:dd
dense_15_102021:d!
dense_16_102043:dd
dense_16_102045:d!
dense_17_102067:dd
dense_17_102069:d!
dense_18_102091:dd
dense_18_102093:d!
dense_19_102115:dd
dense_19_102117:d!
dense_20_102132:d

dense_20_102134:

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�dense_2/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101638�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_101659dense_101661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101658�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_101683dense_1_101685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101682�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_101707dense_2_101709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101706�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_101731dense_3_101733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_101730�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_101755dense_4_101757*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_101754�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_101779dense_5_101781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_101778�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_101803dense_6_101805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_101802�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_101827dense_7_101829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_101826�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_101851dense_8_101853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_101850�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_101875dense_9_101877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_101874�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_101899dense_10_101901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_101898�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_101923dense_11_101925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_101922�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_101947dense_12_101949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_101946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_101971dense_13_101973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_101970�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_101995dense_14_101997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_101994�
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_102019dense_15_102021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_102018�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_102043dense_16_102045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_102042�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_102067dense_17_102069*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_102066�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_102091dense_18_102093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_102090�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_102115dense_19_102117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_102114�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_102132dense_20_102134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_102131x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�j
�
F__inference_sequential_layer_call_and_return_conditional_losses_102928
flatten_input
dense_102822:	�d
dense_102824:d 
dense_1_102827:dd
dense_1_102829:d 
dense_2_102832:dd
dense_2_102834:d 
dense_3_102837:dd
dense_3_102839:d 
dense_4_102842:dd
dense_4_102844:d 
dense_5_102847:dd
dense_5_102849:d 
dense_6_102852:dd
dense_6_102854:d 
dense_7_102857:dd
dense_7_102859:d 
dense_8_102862:dd
dense_8_102864:d 
dense_9_102867:dd
dense_9_102869:d!
dense_10_102872:dd
dense_10_102874:d!
dense_11_102877:dd
dense_11_102879:d!
dense_12_102882:dd
dense_12_102884:d!
dense_13_102887:dd
dense_13_102889:d!
dense_14_102892:dd
dense_14_102894:d!
dense_15_102897:dd
dense_15_102899:d!
dense_16_102902:dd
dense_16_102904:d!
dense_17_102907:dd
dense_17_102909:d!
dense_18_102912:dd
dense_18_102914:d!
dense_19_102917:dd
dense_19_102919:d!
dense_20_102922:d

dense_20_102924:

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�dense_2/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101638�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_102822dense_102824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101658�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_102827dense_1_102829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_101682�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_102832dense_2_102834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101706�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_102837dense_3_102839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_101730�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_102842dense_4_102844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_101754�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_102847dense_5_102849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_101778�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_102852dense_6_102854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_101802�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_102857dense_7_102859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_101826�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_102862dense_8_102864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_101850�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_102867dense_9_102869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_101874�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_102872dense_10_102874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_101898�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_102877dense_11_102879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_101922�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_102882dense_12_102884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_101946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_102887dense_13_102889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_101970�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_102892dense_14_102894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_101994�
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_102897dense_15_102899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_102018�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_102902dense_16_102904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_102042�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_102907dense_17_102909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_102066�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_102912dense_18_102914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_102090�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_102917dense_19_102919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_102114�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_102922dense_20_102924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_102131x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:^ Z
/
_output_shapes
:���������  
'
_user_specified_nameflatten_input
�
�
#__inference_internal_grad_fn_106491
result_grads_0
result_grads_1 
mul_sequential_dense_16_beta#
mul_sequential_dense_16_biasadd
identity�
mulMulmul_sequential_dense_16_betamul_sequential_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d}
mul_1Mulmul_sequential_dense_16_betamul_sequential_dense_16_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
#__inference_internal_grad_fn_106365
result_grads_0
result_grads_1
mul_sequential_dense_9_beta"
mul_sequential_dense_9_biasadd
identity�
mulMulmul_sequential_dense_9_betamul_sequential_dense_9_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������d{
mul_1Mulmul_sequential_dense_9_betamul_sequential_dense_9_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
(__inference_dense_8_layer_call_fn_104135

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_101850o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_12_layer_call_fn_104243

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_101946o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_dense_17_layer_call_fn_104378

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_102066o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
z
#__inference_internal_grad_fn_104979
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������dU
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
)__inference_dense_18_layer_call_fn_104405

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_102090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�	
+__inference_sequential_layer_call_fn_103224

inputs
unknown:	�d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:dd

unknown_12:d

unknown_13:dd

unknown_14:d

unknown_15:dd

unknown_16:d

unknown_17:dd

unknown_18:d

unknown_19:dd

unknown_20:d

unknown_21:dd

unknown_22:d

unknown_23:dd

unknown_24:d

unknown_25:dd

unknown_26:d

unknown_27:dd

unknown_28:d

unknown_29:dd

unknown_30:d

unknown_31:dd

unknown_32:d

unknown_33:dd

unknown_34:d

unknown_35:dd

unknown_36:d

unknown_37:dd

unknown_38:d

unknown_39:d


unknown_40:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_102138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesq
o:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
#__inference_internal_grad_fn_105519
result_grads_0
result_grads_1
mul_dense_2_beta
mul_dense_2_biasadd
identityt
mulMulmul_dense_2_betamul_dense_2_biasadd^result_grads_0*
T0*'
_output_shapes
:���������dM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������de
mul_1Mulmul_dense_2_betamul_dense_2_biasadd*
T0*'
_output_shapes
:���������dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������dR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������dJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������dT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������dY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������dQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*N
_input_shapes=
;:���������d:���������d: :���������d:W S
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������d
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������d
�
�
(__inference_dense_2_layer_call_fn_103973

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_101706o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_104763CustomGradient-104442<
#__inference_internal_grad_fn_104781CustomGradient-102106<
#__inference_internal_grad_fn_104799CustomGradient-104415<
#__inference_internal_grad_fn_104817CustomGradient-102082<
#__inference_internal_grad_fn_104835CustomGradient-104388<
#__inference_internal_grad_fn_104853CustomGradient-102058<
#__inference_internal_grad_fn_104871CustomGradient-104361<
#__inference_internal_grad_fn_104889CustomGradient-102034<
#__inference_internal_grad_fn_104907CustomGradient-104334<
#__inference_internal_grad_fn_104925CustomGradient-102010<
#__inference_internal_grad_fn_104943CustomGradient-104307<
#__inference_internal_grad_fn_104961CustomGradient-101986<
#__inference_internal_grad_fn_104979CustomGradient-104280<
#__inference_internal_grad_fn_104997CustomGradient-101962<
#__inference_internal_grad_fn_105015CustomGradient-104253<
#__inference_internal_grad_fn_105033CustomGradient-101938<
#__inference_internal_grad_fn_105051CustomGradient-104226<
#__inference_internal_grad_fn_105069CustomGradient-101914<
#__inference_internal_grad_fn_105087CustomGradient-104199<
#__inference_internal_grad_fn_105105CustomGradient-101890<
#__inference_internal_grad_fn_105123CustomGradient-104172<
#__inference_internal_grad_fn_105141CustomGradient-101866<
#__inference_internal_grad_fn_105159CustomGradient-104145<
#__inference_internal_grad_fn_105177CustomGradient-101842<
#__inference_internal_grad_fn_105195CustomGradient-104118<
#__inference_internal_grad_fn_105213CustomGradient-101818<
#__inference_internal_grad_fn_105231CustomGradient-104091<
#__inference_internal_grad_fn_105249CustomGradient-101794<
#__inference_internal_grad_fn_105267CustomGradient-104064<
#__inference_internal_grad_fn_105285CustomGradient-101770<
#__inference_internal_grad_fn_105303CustomGradient-104037<
#__inference_internal_grad_fn_105321CustomGradient-101746<
#__inference_internal_grad_fn_105339CustomGradient-104010<
#__inference_internal_grad_fn_105357CustomGradient-101722<
#__inference_internal_grad_fn_105375CustomGradient-103983<
#__inference_internal_grad_fn_105393CustomGradient-101698<
#__inference_internal_grad_fn_105411CustomGradient-103956<
#__inference_internal_grad_fn_105429CustomGradient-101674<
#__inference_internal_grad_fn_105447CustomGradient-103929<
#__inference_internal_grad_fn_105465CustomGradient-101650<
#__inference_internal_grad_fn_105483CustomGradient-103325<
#__inference_internal_grad_fn_105501CustomGradient-103339<
#__inference_internal_grad_fn_105519CustomGradient-103353<
#__inference_internal_grad_fn_105537CustomGradient-103367<
#__inference_internal_grad_fn_105555CustomGradient-103381<
#__inference_internal_grad_fn_105573CustomGradient-103395<
#__inference_internal_grad_fn_105591CustomGradient-103409<
#__inference_internal_grad_fn_105609CustomGradient-103423<
#__inference_internal_grad_fn_105627CustomGradient-103437<
#__inference_internal_grad_fn_105645CustomGradient-103451<
#__inference_internal_grad_fn_105663CustomGradient-103465<
#__inference_internal_grad_fn_105681CustomGradient-103479<
#__inference_internal_grad_fn_105699CustomGradient-103493<
#__inference_internal_grad_fn_105717CustomGradient-103507<
#__inference_internal_grad_fn_105735CustomGradient-103521<
#__inference_internal_grad_fn_105753CustomGradient-103535<
#__inference_internal_grad_fn_105771CustomGradient-103549<
#__inference_internal_grad_fn_105789CustomGradient-103563<
#__inference_internal_grad_fn_105807CustomGradient-103577<
#__inference_internal_grad_fn_105825CustomGradient-103591<
#__inference_internal_grad_fn_105843CustomGradient-103618<
#__inference_internal_grad_fn_105861CustomGradient-103632<
#__inference_internal_grad_fn_105879CustomGradient-103646<
#__inference_internal_grad_fn_105897CustomGradient-103660<
#__inference_internal_grad_fn_105915CustomGradient-103674<
#__inference_internal_grad_fn_105933CustomGradient-103688<
#__inference_internal_grad_fn_105951CustomGradient-103702<
#__inference_internal_grad_fn_105969CustomGradient-103716<
#__inference_internal_grad_fn_105987CustomGradient-103730<
#__inference_internal_grad_fn_106005CustomGradient-103744<
#__inference_internal_grad_fn_106023CustomGradient-103758<
#__inference_internal_grad_fn_106041CustomGradient-103772<
#__inference_internal_grad_fn_106059CustomGradient-103786<
#__inference_internal_grad_fn_106077CustomGradient-103800<
#__inference_internal_grad_fn_106095CustomGradient-103814<
#__inference_internal_grad_fn_106113CustomGradient-103828<
#__inference_internal_grad_fn_106131CustomGradient-103842<
#__inference_internal_grad_fn_106149CustomGradient-103856<
#__inference_internal_grad_fn_106167CustomGradient-103870<
#__inference_internal_grad_fn_106185CustomGradient-103884<
#__inference_internal_grad_fn_106203CustomGradient-101344<
#__inference_internal_grad_fn_106221CustomGradient-101358<
#__inference_internal_grad_fn_106239CustomGradient-101372<
#__inference_internal_grad_fn_106257CustomGradient-101386<
#__inference_internal_grad_fn_106275CustomGradient-101400<
#__inference_internal_grad_fn_106293CustomGradient-101414<
#__inference_internal_grad_fn_106311CustomGradient-101428<
#__inference_internal_grad_fn_106329CustomGradient-101442<
#__inference_internal_grad_fn_106347CustomGradient-101456<
#__inference_internal_grad_fn_106365CustomGradient-101470<
#__inference_internal_grad_fn_106383CustomGradient-101484<
#__inference_internal_grad_fn_106401CustomGradient-101498<
#__inference_internal_grad_fn_106419CustomGradient-101512<
#__inference_internal_grad_fn_106437CustomGradient-101526<
#__inference_internal_grad_fn_106455CustomGradient-101540<
#__inference_internal_grad_fn_106473CustomGradient-101554<
#__inference_internal_grad_fn_106491CustomGradient-101568<
#__inference_internal_grad_fn_106509CustomGradient-101582<
#__inference_internal_grad_fn_106527CustomGradient-101596<
#__inference_internal_grad_fn_106545CustomGradient-101610"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O
flatten_input>
serving_default_flatten_input:0���������  <
dense_200
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer_with_weights-13
layer-14
layer_with_weights-14
layer-15
layer_with_weights-15
layer-16
layer_with_weights-16
layer-17
layer_with_weights-17
layer-18
layer_with_weights-18
layer-19
layer_with_weights-19
layer-20
layer_with_weights-20
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
,0
-1
42
53
<4
=5
D6
E7
L8
M9
T10
U11
\12
]13
d14
e15
l16
m17
t18
u19
|20
}21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41"
trackable_list_wrapper
�
,0
-1
42
53
<4
=5
D6
E7
L8
M9
T10
U11
\12
]13
d14
e15
l16
m17
t18
u19
|20
}21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
+__inference_sequential_layer_call_fn_102225
+__inference_sequential_layer_call_fn_103224
+__inference_sequential_layer_call_fn_103313
+__inference_sequential_layer_call_fn_102818�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
F__inference_sequential_layer_call_and_return_conditional_losses_103606
F__inference_sequential_layer_call_and_return_conditional_losses_103899
F__inference_sequential_layer_call_and_return_conditional_losses_102928
F__inference_sequential_layer_call_and_return_conditional_losses_103038�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_101625flatten_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate
�momentum_cache,m�-m�4m�5m�<m�=m�Dm�Em�Lm�Mm�Tm�Um�\m�]m�dm�em�lm�mm�tm�um�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�,v�-v�4v�5v�<v�=v�Dv�Ev�Lv�Mv�Tv�Uv�\v�]v�dv�ev�lv�mv�tv�uv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_103904�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_103910�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_103919�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_103937�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�d2dense/kernel
:d2
dense/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_103946�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_103964�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_1/kernel
:d2dense_1/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_103973�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_103991�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_2/kernel
:d2dense_2/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_104000�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_104018�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_3/kernel
:d2dense_3/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_4_layer_call_fn_104027�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_4_layer_call_and_return_conditional_losses_104045�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_4/kernel
:d2dense_4/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_5_layer_call_fn_104054�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_5_layer_call_and_return_conditional_losses_104072�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_5/kernel
:d2dense_5/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_6_layer_call_fn_104081�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_6_layer_call_and_return_conditional_losses_104099�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_6/kernel
:d2dense_6/bias
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_7_layer_call_fn_104108�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_7_layer_call_and_return_conditional_losses_104126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_7/kernel
:d2dense_7/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_8_layer_call_fn_104135�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_8_layer_call_and_return_conditional_losses_104153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_8/kernel
:d2dense_8/bias
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_9_layer_call_fn_104162�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_9_layer_call_and_return_conditional_losses_104180�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :dd2dense_9/kernel
:d2dense_9/bias
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_10_layer_call_fn_104189�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_10_layer_call_and_return_conditional_losses_104207�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_10/kernel
:d2dense_10/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_11_layer_call_fn_104216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_11_layer_call_and_return_conditional_losses_104234�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_11/kernel
:d2dense_11/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_12_layer_call_fn_104243�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_12_layer_call_and_return_conditional_losses_104261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_12/kernel
:d2dense_12/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_13_layer_call_fn_104270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_13_layer_call_and_return_conditional_losses_104288�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_13/kernel
:d2dense_13/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_14_layer_call_fn_104297�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_14_layer_call_and_return_conditional_losses_104315�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_14/kernel
:d2dense_14/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_15_layer_call_fn_104324�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_15_layer_call_and_return_conditional_losses_104342�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_15/kernel
:d2dense_15/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_16_layer_call_fn_104351�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_16_layer_call_and_return_conditional_losses_104369�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_16/kernel
:d2dense_16/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_17_layer_call_fn_104378�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_17_layer_call_and_return_conditional_losses_104396�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_17/kernel
:d2dense_17/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_18_layer_call_fn_104405�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_18_layer_call_and_return_conditional_losses_104423�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_18/kernel
:d2dense_18/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_19_layer_call_fn_104432�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_19_layer_call_and_return_conditional_losses_104450�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:dd2dense_19/kernel
:d2dense_19/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_20_layer_call_fn_104459�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_20_layer_call_and_return_conditional_losses_104470�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:d
2dense_20/kernel
:
2dense_20/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_sequential_layer_call_fn_102225flatten_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_103224inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_103313inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_102818flatten_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_103606inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_103899inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_102928flatten_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_103038flatten_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
�B�
$__inference_signature_wrapper_103135flatten_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_flatten_layer_call_fn_103904inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_103910inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_103919inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_103937inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_103946inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_103964inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_103973inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_103991inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_104000inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_104018inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_4_layer_call_fn_104027inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_4_layer_call_and_return_conditional_losses_104045inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_5_layer_call_fn_104054inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_5_layer_call_and_return_conditional_losses_104072inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_6_layer_call_fn_104081inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_6_layer_call_and_return_conditional_losses_104099inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_7_layer_call_fn_104108inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_7_layer_call_and_return_conditional_losses_104126inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_8_layer_call_fn_104135inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_8_layer_call_and_return_conditional_losses_104153inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_9_layer_call_fn_104162inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_9_layer_call_and_return_conditional_losses_104180inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_10_layer_call_fn_104189inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_10_layer_call_and_return_conditional_losses_104207inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_11_layer_call_fn_104216inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_11_layer_call_and_return_conditional_losses_104234inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_12_layer_call_fn_104243inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_12_layer_call_and_return_conditional_losses_104261inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_13_layer_call_fn_104270inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_13_layer_call_and_return_conditional_losses_104288inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_14_layer_call_fn_104297inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_14_layer_call_and_return_conditional_losses_104315inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_15_layer_call_fn_104324inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_15_layer_call_and_return_conditional_losses_104342inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_16_layer_call_fn_104351inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_16_layer_call_and_return_conditional_losses_104369inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_17_layer_call_fn_104378inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_17_layer_call_and_return_conditional_losses_104396inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_18_layer_call_fn_104405inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_18_layer_call_and_return_conditional_losses_104423inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_19_layer_call_fn_104432inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_19_layer_call_and_return_conditional_losses_104450inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_20_layer_call_fn_104459inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_20_layer_call_and_return_conditional_losses_104470inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
%:#	�d2Nadam/dense/kernel/m
:d2Nadam/dense/bias/m
&:$dd2Nadam/dense_1/kernel/m
 :d2Nadam/dense_1/bias/m
&:$dd2Nadam/dense_2/kernel/m
 :d2Nadam/dense_2/bias/m
&:$dd2Nadam/dense_3/kernel/m
 :d2Nadam/dense_3/bias/m
&:$dd2Nadam/dense_4/kernel/m
 :d2Nadam/dense_4/bias/m
&:$dd2Nadam/dense_5/kernel/m
 :d2Nadam/dense_5/bias/m
&:$dd2Nadam/dense_6/kernel/m
 :d2Nadam/dense_6/bias/m
&:$dd2Nadam/dense_7/kernel/m
 :d2Nadam/dense_7/bias/m
&:$dd2Nadam/dense_8/kernel/m
 :d2Nadam/dense_8/bias/m
&:$dd2Nadam/dense_9/kernel/m
 :d2Nadam/dense_9/bias/m
':%dd2Nadam/dense_10/kernel/m
!:d2Nadam/dense_10/bias/m
':%dd2Nadam/dense_11/kernel/m
!:d2Nadam/dense_11/bias/m
':%dd2Nadam/dense_12/kernel/m
!:d2Nadam/dense_12/bias/m
':%dd2Nadam/dense_13/kernel/m
!:d2Nadam/dense_13/bias/m
':%dd2Nadam/dense_14/kernel/m
!:d2Nadam/dense_14/bias/m
':%dd2Nadam/dense_15/kernel/m
!:d2Nadam/dense_15/bias/m
':%dd2Nadam/dense_16/kernel/m
!:d2Nadam/dense_16/bias/m
':%dd2Nadam/dense_17/kernel/m
!:d2Nadam/dense_17/bias/m
':%dd2Nadam/dense_18/kernel/m
!:d2Nadam/dense_18/bias/m
':%dd2Nadam/dense_19/kernel/m
!:d2Nadam/dense_19/bias/m
':%d
2Nadam/dense_20/kernel/m
!:
2Nadam/dense_20/bias/m
%:#	�d2Nadam/dense/kernel/v
:d2Nadam/dense/bias/v
&:$dd2Nadam/dense_1/kernel/v
 :d2Nadam/dense_1/bias/v
&:$dd2Nadam/dense_2/kernel/v
 :d2Nadam/dense_2/bias/v
&:$dd2Nadam/dense_3/kernel/v
 :d2Nadam/dense_3/bias/v
&:$dd2Nadam/dense_4/kernel/v
 :d2Nadam/dense_4/bias/v
&:$dd2Nadam/dense_5/kernel/v
 :d2Nadam/dense_5/bias/v
&:$dd2Nadam/dense_6/kernel/v
 :d2Nadam/dense_6/bias/v
&:$dd2Nadam/dense_7/kernel/v
 :d2Nadam/dense_7/bias/v
&:$dd2Nadam/dense_8/kernel/v
 :d2Nadam/dense_8/bias/v
&:$dd2Nadam/dense_9/kernel/v
 :d2Nadam/dense_9/bias/v
':%dd2Nadam/dense_10/kernel/v
!:d2Nadam/dense_10/bias/v
':%dd2Nadam/dense_11/kernel/v
!:d2Nadam/dense_11/bias/v
':%dd2Nadam/dense_12/kernel/v
!:d2Nadam/dense_12/bias/v
':%dd2Nadam/dense_13/kernel/v
!:d2Nadam/dense_13/bias/v
':%dd2Nadam/dense_14/kernel/v
!:d2Nadam/dense_14/bias/v
':%dd2Nadam/dense_15/kernel/v
!:d2Nadam/dense_15/bias/v
':%dd2Nadam/dense_16/kernel/v
!:d2Nadam/dense_16/bias/v
':%dd2Nadam/dense_17/kernel/v
!:d2Nadam/dense_17/bias/v
':%dd2Nadam/dense_18/kernel/v
!:d2Nadam/dense_18/bias/v
':%dd2Nadam/dense_19/kernel/v
!:d2Nadam/dense_19/bias/v
':%d
2Nadam/dense_20/kernel/v
!:
2Nadam/dense_20/bias/v
PbN
beta:0D__inference_dense_19_layer_call_and_return_conditional_losses_104450
SbQ
	BiasAdd:0D__inference_dense_19_layer_call_and_return_conditional_losses_104450
PbN
beta:0D__inference_dense_19_layer_call_and_return_conditional_losses_102114
SbQ
	BiasAdd:0D__inference_dense_19_layer_call_and_return_conditional_losses_102114
PbN
beta:0D__inference_dense_18_layer_call_and_return_conditional_losses_104423
SbQ
	BiasAdd:0D__inference_dense_18_layer_call_and_return_conditional_losses_104423
PbN
beta:0D__inference_dense_18_layer_call_and_return_conditional_losses_102090
SbQ
	BiasAdd:0D__inference_dense_18_layer_call_and_return_conditional_losses_102090
PbN
beta:0D__inference_dense_17_layer_call_and_return_conditional_losses_104396
SbQ
	BiasAdd:0D__inference_dense_17_layer_call_and_return_conditional_losses_104396
PbN
beta:0D__inference_dense_17_layer_call_and_return_conditional_losses_102066
SbQ
	BiasAdd:0D__inference_dense_17_layer_call_and_return_conditional_losses_102066
PbN
beta:0D__inference_dense_16_layer_call_and_return_conditional_losses_104369
SbQ
	BiasAdd:0D__inference_dense_16_layer_call_and_return_conditional_losses_104369
PbN
beta:0D__inference_dense_16_layer_call_and_return_conditional_losses_102042
SbQ
	BiasAdd:0D__inference_dense_16_layer_call_and_return_conditional_losses_102042
PbN
beta:0D__inference_dense_15_layer_call_and_return_conditional_losses_104342
SbQ
	BiasAdd:0D__inference_dense_15_layer_call_and_return_conditional_losses_104342
PbN
beta:0D__inference_dense_15_layer_call_and_return_conditional_losses_102018
SbQ
	BiasAdd:0D__inference_dense_15_layer_call_and_return_conditional_losses_102018
PbN
beta:0D__inference_dense_14_layer_call_and_return_conditional_losses_104315
SbQ
	BiasAdd:0D__inference_dense_14_layer_call_and_return_conditional_losses_104315
PbN
beta:0D__inference_dense_14_layer_call_and_return_conditional_losses_101994
SbQ
	BiasAdd:0D__inference_dense_14_layer_call_and_return_conditional_losses_101994
PbN
beta:0D__inference_dense_13_layer_call_and_return_conditional_losses_104288
SbQ
	BiasAdd:0D__inference_dense_13_layer_call_and_return_conditional_losses_104288
PbN
beta:0D__inference_dense_13_layer_call_and_return_conditional_losses_101970
SbQ
	BiasAdd:0D__inference_dense_13_layer_call_and_return_conditional_losses_101970
PbN
beta:0D__inference_dense_12_layer_call_and_return_conditional_losses_104261
SbQ
	BiasAdd:0D__inference_dense_12_layer_call_and_return_conditional_losses_104261
PbN
beta:0D__inference_dense_12_layer_call_and_return_conditional_losses_101946
SbQ
	BiasAdd:0D__inference_dense_12_layer_call_and_return_conditional_losses_101946
PbN
beta:0D__inference_dense_11_layer_call_and_return_conditional_losses_104234
SbQ
	BiasAdd:0D__inference_dense_11_layer_call_and_return_conditional_losses_104234
PbN
beta:0D__inference_dense_11_layer_call_and_return_conditional_losses_101922
SbQ
	BiasAdd:0D__inference_dense_11_layer_call_and_return_conditional_losses_101922
PbN
beta:0D__inference_dense_10_layer_call_and_return_conditional_losses_104207
SbQ
	BiasAdd:0D__inference_dense_10_layer_call_and_return_conditional_losses_104207
PbN
beta:0D__inference_dense_10_layer_call_and_return_conditional_losses_101898
SbQ
	BiasAdd:0D__inference_dense_10_layer_call_and_return_conditional_losses_101898
ObM
beta:0C__inference_dense_9_layer_call_and_return_conditional_losses_104180
RbP
	BiasAdd:0C__inference_dense_9_layer_call_and_return_conditional_losses_104180
ObM
beta:0C__inference_dense_9_layer_call_and_return_conditional_losses_101874
RbP
	BiasAdd:0C__inference_dense_9_layer_call_and_return_conditional_losses_101874
ObM
beta:0C__inference_dense_8_layer_call_and_return_conditional_losses_104153
RbP
	BiasAdd:0C__inference_dense_8_layer_call_and_return_conditional_losses_104153
ObM
beta:0C__inference_dense_8_layer_call_and_return_conditional_losses_101850
RbP
	BiasAdd:0C__inference_dense_8_layer_call_and_return_conditional_losses_101850
ObM
beta:0C__inference_dense_7_layer_call_and_return_conditional_losses_104126
RbP
	BiasAdd:0C__inference_dense_7_layer_call_and_return_conditional_losses_104126
ObM
beta:0C__inference_dense_7_layer_call_and_return_conditional_losses_101826
RbP
	BiasAdd:0C__inference_dense_7_layer_call_and_return_conditional_losses_101826
ObM
beta:0C__inference_dense_6_layer_call_and_return_conditional_losses_104099
RbP
	BiasAdd:0C__inference_dense_6_layer_call_and_return_conditional_losses_104099
ObM
beta:0C__inference_dense_6_layer_call_and_return_conditional_losses_101802
RbP
	BiasAdd:0C__inference_dense_6_layer_call_and_return_conditional_losses_101802
ObM
beta:0C__inference_dense_5_layer_call_and_return_conditional_losses_104072
RbP
	BiasAdd:0C__inference_dense_5_layer_call_and_return_conditional_losses_104072
ObM
beta:0C__inference_dense_5_layer_call_and_return_conditional_losses_101778
RbP
	BiasAdd:0C__inference_dense_5_layer_call_and_return_conditional_losses_101778
ObM
beta:0C__inference_dense_4_layer_call_and_return_conditional_losses_104045
RbP
	BiasAdd:0C__inference_dense_4_layer_call_and_return_conditional_losses_104045
ObM
beta:0C__inference_dense_4_layer_call_and_return_conditional_losses_101754
RbP
	BiasAdd:0C__inference_dense_4_layer_call_and_return_conditional_losses_101754
ObM
beta:0C__inference_dense_3_layer_call_and_return_conditional_losses_104018
RbP
	BiasAdd:0C__inference_dense_3_layer_call_and_return_conditional_losses_104018
ObM
beta:0C__inference_dense_3_layer_call_and_return_conditional_losses_101730
RbP
	BiasAdd:0C__inference_dense_3_layer_call_and_return_conditional_losses_101730
ObM
beta:0C__inference_dense_2_layer_call_and_return_conditional_losses_103991
RbP
	BiasAdd:0C__inference_dense_2_layer_call_and_return_conditional_losses_103991
ObM
beta:0C__inference_dense_2_layer_call_and_return_conditional_losses_101706
RbP
	BiasAdd:0C__inference_dense_2_layer_call_and_return_conditional_losses_101706
ObM
beta:0C__inference_dense_1_layer_call_and_return_conditional_losses_103964
RbP
	BiasAdd:0C__inference_dense_1_layer_call_and_return_conditional_losses_103964
ObM
beta:0C__inference_dense_1_layer_call_and_return_conditional_losses_101682
RbP
	BiasAdd:0C__inference_dense_1_layer_call_and_return_conditional_losses_101682
MbK
beta:0A__inference_dense_layer_call_and_return_conditional_losses_103937
PbN
	BiasAdd:0A__inference_dense_layer_call_and_return_conditional_losses_103937
MbK
beta:0A__inference_dense_layer_call_and_return_conditional_losses_101658
PbN
	BiasAdd:0A__inference_dense_layer_call_and_return_conditional_losses_101658
XbV
dense/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_1/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_1/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_2/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_2/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_3/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_3/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_4/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_4/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_5/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_5/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_6/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_6/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_7/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_7/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_8/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_8/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
ZbX
dense_9/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
]b[
dense_9/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_10/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_10/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_11/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_11/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_12/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_12/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_13/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_13/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_14/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_14/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_15/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_15/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_16/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_16/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_17/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_17/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_18/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_18/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
[bY
dense_19/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
^b\
dense_19/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103606
XbV
dense/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_1/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_1/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_2/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_2/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_3/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_3/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_4/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_4/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_5/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_5/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_6/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_6/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_7/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_7/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_8/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_8/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
ZbX
dense_9/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
]b[
dense_9/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_10/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_10/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_11/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_11/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_12/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_12/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_13/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_13/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_14/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_14/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_15/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_15/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_16/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_16/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_17/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_17/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_18/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_18/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
[bY
dense_19/beta:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
^b\
dense_19/BiasAdd:0F__inference_sequential_layer_call_and_return_conditional_losses_103899
>b<
sequential/dense/beta:0!__inference__wrapped_model_101625
Ab?
sequential/dense/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_1/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_1/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_2/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_2/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_3/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_3/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_4/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_4/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_5/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_5/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_6/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_6/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_7/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_7/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_8/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_8/BiasAdd:0!__inference__wrapped_model_101625
@b>
sequential/dense_9/beta:0!__inference__wrapped_model_101625
CbA
sequential/dense_9/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_10/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_10/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_11/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_11/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_12/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_12/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_13/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_13/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_14/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_14/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_15/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_15/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_16/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_16/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_17/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_17/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_18/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_18/BiasAdd:0!__inference__wrapped_model_101625
Ab?
sequential/dense_19/beta:0!__inference__wrapped_model_101625
DbB
sequential/dense_19/BiasAdd:0!__inference__wrapped_model_101625�
!__inference__wrapped_model_101625�>,-45<=DELMTU\]delmtu|}��������������������>�;
4�1
/�,
flatten_input���������  
� "3�0
.
dense_20"�
dense_20���������
�
D__inference_dense_10_layer_call_and_return_conditional_losses_104207\|}/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� |
)__inference_dense_10_layer_call_fn_104189O|}/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_11_layer_call_and_return_conditional_losses_104234^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_11_layer_call_fn_104216Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_12_layer_call_and_return_conditional_losses_104261^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_12_layer_call_fn_104243Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_13_layer_call_and_return_conditional_losses_104288^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_13_layer_call_fn_104270Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_14_layer_call_and_return_conditional_losses_104315^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_14_layer_call_fn_104297Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_15_layer_call_and_return_conditional_losses_104342^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_15_layer_call_fn_104324Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_16_layer_call_and_return_conditional_losses_104369^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_16_layer_call_fn_104351Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_17_layer_call_and_return_conditional_losses_104396^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_17_layer_call_fn_104378Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_18_layer_call_and_return_conditional_losses_104423^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_18_layer_call_fn_104405Q��/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_19_layer_call_and_return_conditional_losses_104450^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� ~
)__inference_dense_19_layer_call_fn_104432Q��/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_1_layer_call_and_return_conditional_losses_103964\45/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_1_layer_call_fn_103946O45/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_20_layer_call_and_return_conditional_losses_104470^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������

� ~
)__inference_dense_20_layer_call_fn_104459Q��/�,
%�"
 �
inputs���������d
� "����������
�
C__inference_dense_2_layer_call_and_return_conditional_losses_103991\<=/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_2_layer_call_fn_103973O<=/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_3_layer_call_and_return_conditional_losses_104018\DE/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_3_layer_call_fn_104000ODE/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_4_layer_call_and_return_conditional_losses_104045\LM/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_4_layer_call_fn_104027OLM/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_5_layer_call_and_return_conditional_losses_104072\TU/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_5_layer_call_fn_104054OTU/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_6_layer_call_and_return_conditional_losses_104099\\]/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_6_layer_call_fn_104081O\]/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_7_layer_call_and_return_conditional_losses_104126\de/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_7_layer_call_fn_104108Ode/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_8_layer_call_and_return_conditional_losses_104153\lm/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_8_layer_call_fn_104135Olm/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_9_layer_call_and_return_conditional_losses_104180\tu/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_9_layer_call_fn_104162Otu/�,
%�"
 �
inputs���������d
� "����������d�
A__inference_dense_layer_call_and_return_conditional_losses_103937],-0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� z
&__inference_dense_layer_call_fn_103919P,-0�-
&�#
!�
inputs����������
� "����������d�
C__inference_flatten_layer_call_and_return_conditional_losses_103910a7�4
-�*
(�%
inputs���������  
� "&�#
�
0����������
� �
(__inference_flatten_layer_call_fn_103904T7�4
-�*
(�%
inputs���������  
� "������������
#__inference_internal_grad_fn_104763���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104781���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104799���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104817���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104835���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104853���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104871���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104889���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104907���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104925���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104943���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104961���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104979���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_104997���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105015���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105033���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105051���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105069���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105087���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105105���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105123���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105141���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105159���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105177���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105195���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105213���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105231���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105249���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105267���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105285���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105303���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105321���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105339���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105357���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105375���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105393���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105411���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105429���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105447���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105465���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105483���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105501���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105519���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105537���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105555���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105573���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105591���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105609���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105627���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105645���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105663���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105681���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105699���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105717���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105735���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105753���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105771���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105789���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105807���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105825���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105843���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105861���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105879���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105897���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105915���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105933���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105951���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105969���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_105987���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106005���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106023���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106041���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106059���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106077���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106095���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106113���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106131���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106149���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106167���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106185���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106203���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106221���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106239���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106257���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106275���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106293���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106311���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106329���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106347���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106365���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106383���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106401���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106419���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106437���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106455���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106473���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106491���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106509���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106527���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
#__inference_internal_grad_fn_106545���e�b
[�X

 
(�%
result_grads_0���������d
(�%
result_grads_1���������d
� "$�!

 
�
1���������d�
F__inference_sequential_layer_call_and_return_conditional_losses_102928�>,-45<=DELMTU\]delmtu|}��������������������F�C
<�9
/�,
flatten_input���������  
p 

 
� "%�"
�
0���������

� �
F__inference_sequential_layer_call_and_return_conditional_losses_103038�>,-45<=DELMTU\]delmtu|}��������������������F�C
<�9
/�,
flatten_input���������  
p

 
� "%�"
�
0���������

� �
F__inference_sequential_layer_call_and_return_conditional_losses_103606�>,-45<=DELMTU\]delmtu|}��������������������?�<
5�2
(�%
inputs���������  
p 

 
� "%�"
�
0���������

� �
F__inference_sequential_layer_call_and_return_conditional_losses_103899�>,-45<=DELMTU\]delmtu|}��������������������?�<
5�2
(�%
inputs���������  
p

 
� "%�"
�
0���������

� �
+__inference_sequential_layer_call_fn_102225�>,-45<=DELMTU\]delmtu|}��������������������F�C
<�9
/�,
flatten_input���������  
p 

 
� "����������
�
+__inference_sequential_layer_call_fn_102818�>,-45<=DELMTU\]delmtu|}��������������������F�C
<�9
/�,
flatten_input���������  
p

 
� "����������
�
+__inference_sequential_layer_call_fn_103224�>,-45<=DELMTU\]delmtu|}��������������������?�<
5�2
(�%
inputs���������  
p 

 
� "����������
�
+__inference_sequential_layer_call_fn_103313�>,-45<=DELMTU\]delmtu|}��������������������?�<
5�2
(�%
inputs���������  
p

 
� "����������
�
$__inference_signature_wrapper_103135�>,-45<=DELMTU\]delmtu|}��������������������O�L
� 
E�B
@
flatten_input/�,
flatten_input���������  "3�0
.
dense_20"�
dense_20���������
