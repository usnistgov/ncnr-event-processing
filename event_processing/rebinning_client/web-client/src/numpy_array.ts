export class NumpyArray {
  data: string
  dtype: string
  shape: number[]

  constructor({data, dtype, shape} : {data: string, dtype: string, shape: number[]}) {
    this.data = data;
    this.dtype = dtype;
    this.shape = shape;
  }

  to_typed_array() {
    const accessor = getAccessor(this.dtype);
    const bytes = base64ToBytes(this.data);
    return new accessor(bytes.buffer);
  }

  to_array() {
    const num_array = [...this.to_typed_array()].map(Number);
    return create_nested_array(num_array, this.shape)
  }


  from_array(array_in: number[]) {
    // only handles 1-d arrays for now, and con

  }
}

function map_reverse<Key, Value>(map: Map<Key, Value>): Map<Value, Key> {
  return new Map(Array.from(map.entries()).map(([k, v]) => [v, k]));
}

const int_fmts = new Map([[1, 'b'], [2, 'h'], [4, 'i'], [8, 'q']]);
const float_fmts = new Map([[2, 'e'], [4, 'f'], [8, 'd']]);
const fmts_float = map_reverse(float_fmts);
const fmts_int = map_reverse(int_fmts);


function getAccessor(dtype_str: string): TypedArrayConstructor {
  const match = dtype_str.match(/^([<>|]?)([bhiqefdsBHIQS])([0-9]*)$/);
  if (match == null) {
    throw dtype_str + " is not a recognized dtype"
  }
  const [full, endianness, typestr, length] = match;
  const lower_typestr = typestr.toLowerCase();
  if (fmts_int.has(lower_typestr)) {
    const size = (fmts_int.get(lower_typestr) as number);
    const signed = (lower_typestr == typestr);
    if (size === 8) {
      return (signed) ? BigInt64Array : BigUint64Array;
    }
    else if (size === 4) {
      return (signed) ? Int32Array : Uint32Array;
    }
    else if (size === 2) {
      return (signed) ? Int16Array : Uint16Array;
    }
    else { // size === 1
      return (signed) ? Int8Array : Uint8Array;
    }
  }
  else if (fmts_float.has(lower_typestr)) { // type ==== 1 (float)
    const size = (fmts_float.get(lower_typestr) as number);
    if (size === 8) {
      return Float64Array;
    }
    else if (size === 4) {
      return Float32Array;
    }
    else {
      throw new Error(`Float${size * 8} not supported`);
    }
  }
  else if (lower_typestr === 's') {
    const size = (length === '') ? 4 : parseInt(length, 10);
    const vlen = (length === '');
    alert("don't have accessor for strings yet");
    throw "don't have accessor for strings yet";
  }
  else {
    throw "should never happen"
  }
}

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface NestedArray<T> extends Array<T | NestedArray<T>> { }

function create_nested_array<T>(value: T[], shape: number[]): NestedArray<T> {
  // check that shapes match:
  const total_length = value.length;
  const dims_product = shape.reduce((previous, current) => (previous * current), 1);
  if (total_length !== dims_product) {
    console.warn(`shape product: ${dims_product} does not match length of flattened array: ${total_length}`);
  }

  // Get reshaped output:
  let output: NestedArray<T> = value;
  const subdims = shape.slice(1).reverse();
  for (const dim of subdims) {
    // in each pass, replace input with array of slices of input
    const new_output = [];
    const { length } = output;
    let cursor = 0;
    while (cursor < length) {
      new_output.push(output.slice(cursor, cursor += dim));
    }
    output = new_output;
  }
  return output;
}

type TypedArrayConstructor =
  | Int8ArrayConstructor
  | Uint8ArrayConstructor
  | Uint8ClampedArrayConstructor
  | Int16ArrayConstructor
  | Uint16ArrayConstructor
  | Int32ArrayConstructor
  | Uint32ArrayConstructor
  | BigInt64ArrayConstructor
  | BigUint64ArrayConstructor
  | Float32ArrayConstructor
  | Float64ArrayConstructor;

// from MDN:
function base64ToBytes(base64: string) {
  const binString = atob(base64);
  return Uint8Array.from(binString, (m) => m.codePointAt(0));
}

function bytesToBase64(bytes) {
  const binString = String.fromCodePoint(...bytes);
  return btoa(binString);
}
