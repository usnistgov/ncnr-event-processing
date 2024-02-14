let get_native_littleendian = () => {
  let uInt32 = new Uint32Array([0x11223344]);
  let uInt8 = new Uint8Array(uInt32.buffer);

  if(uInt8[0] === 0x44) {
      return true;
  } else if (uInt8[0] === 0x11) {
      return false;
  } else {
    throw new Error('Maybe mixed-endian?');
  }
};

const is_native_littleendian = get_native_littleendian();

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
    const { endianness, typestr, lengthstr } = parse_dtype(this.dtype);
    const is_littleendian = (endianness !== '>');
    const constructor_key = typestr + lengthstr;
    if (!(constructor_key in typed_array_lookup)) {
      throw new Error(`dtype ${constructor_key} not recognized`);
    }
    
    const constructor = typed_array_lookup[constructor_key];
    const bytes = base64ToBytes(this.data);

    // flip bytes if needed:
    if (is_littleendian != is_native_littleendian && lengthstr) {
      const step = parseInt(lengthstr, 10);
      const view = new DataView(bytes.buffer);
      const buflength = bytes.buffer.byteLength;
      const ntype = number_type[constructor_key];
      const getter = view[`get${ntype}`];
      const setter = view[`set${ntype}`];
      for (let i=0; i<buflength; i+=step) {
        const num_out = getter(i, is_littleendian);
        setter(i, num_out, is_native_littleendian);
      }
    }
    const typed_array = new constructor(bytes.buffer);
    return typed_array;
  }

  to_array() {
    const num_array = [...this.to_typed_array()].map(Number);
    return create_nested_array(num_array, this.shape)
  }

  static from_array(array_in: number[]) {
    // only handles 1-d arrays for now, and automatically uses float64
    const shape = [array_in.length];
    const dtype = `${(is_native_littleendian) ? '<' : '>'}f8`;
    const typed_array = new Float64Array(array_in);
    const data = btoa(new Uint8Array(typed_array.buffer).reduce((d,b) => d + String.fromCharCode(b), ''));
    return new this({data, dtype, shape});
  }
}

const typed_array_lookup = {
  'b': Int8Array,
  'B': Uint8Array,
  'i2': Int16Array,
  'i4': Int32Array,
  'i8': BigInt64Array,
  'u2': Uint16Array,
  'u4': Uint32Array,
  'u8': BigUint64Array,
  'f4': Float32Array,
  'f8': Float64Array,
}

const number_type = {
  'b': 'Int8',
  'B': 'Uint8',
  'i2': 'Int16',
  'i4': 'Int32',
  'i8': 'BigInt64',
  'u2': 'Uint16',
  'u4': 'Uint32',
  'u8': 'BigUint64',
  'f4': 'Float32',
  'f8': 'Float64',
}

function parse_dtype(dtype_str: string) {
  const match = dtype_str.match(/^([<>|]?)([bifuB])([248]?)$/);
  if (match == null) {
    throw dtype_str + " is not a recognized dtype"
  }
  const [full, endianness, typestr, lengthstr] = match;
  return { endianness, typestr, lengthstr };
}


// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface NestedArray<T> extends Array<T | NestedArray<T>> { }

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
