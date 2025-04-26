import pickle
import struct
import tenseal as ts

def send_data(sock, data):
    serialized = pickle.dumps(data)
    length = struct.pack('>I', len(serialized))
    sock.sendall(length + serialized)

def recv_data(sock):
    def recvall(n):
        buf = b''
        while len(buf) < n:
            part = sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("Socket connection closed")
            buf += part
        return buf

    raw_length = recvall(4)
    msg_length = struct.unpack('>I', raw_length)[0]
    data = recvall(msg_length)
    return pickle.loads(data)


def check_alternating_signs(lst):
    for i, val in enumerate(lst):
        if i % 2 == 0:
            if val <= 0:
                return False
        else:
            if val >= 0:
                return False
    return True


def inverse_permutation(perm):
    # Create an array to store the inverse
    inverse = [0] * len(perm)
    
    # Fill the inverse permutation
    for i, p in enumerate(perm):
        inverse[p] = i
    
    return inverse

def writeCkks(ckks_vec, filename):
    ser_ckks_vec = base64.b64encode(ckks_vec)

    with open(filename, 'wb') as f:
        f.write(ser_ckks_vec)

def readCkks(filename):
    with open(filename, 'rb') as f:
        ser_ckks_vec = f.read()
    
    return base64.b64decode(ser_ckks_vec)

def create_context():
    poly_modulus_degree = 2*8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    global_scale= 2**40

    context = ts.context(
                ts.SCHEME_TYPE.CKKS, 
                poly_modulus_degree = poly_modulus_degree,
                coeff_mod_bit_sizes = coeff_mod_bit_sizes
                )
    context.generate_galois_keys()
    context.global_scale = global_scale

    return context