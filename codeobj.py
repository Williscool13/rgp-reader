import struct
def mp_unpack(b,i=0):
    t=b[i]
    if t<=0x7f: return t,i+1
    if t>=0xe0: return t-256,i+1
    if 0x80<=t<=0x8f: return _map(b,i+1,t&0xf)
    if 0x90<=t<=0x9f: return _arr(b,i+1,t&0xf)
    if 0xa0<=t<=0xbf: n=t&0x1f; return b[i+1:i+1+n].decode('utf-8','replace'),i+1+n
    if t==0xc0: return None,i+1
    if t==0xc2: return False,i+1
    if t==0xc3: return True,i+1
    if t in (0xc4,0xc5,0xc6):
        sz={0xc4:1,0xc5:2,0xc6:4}[t]; n=int.from_bytes(b[i+1:i+1+sz],'big'); return bytes(b[i+1+sz:i+1+sz+n]),i+1+sz+n
    if t==0xca: return struct.unpack('>f',b[i+1:i+5])[0],i+5
    if t==0xcb: return struct.unpack('>d',b[i+1:i+9])[0],i+9
    if 0xcc<=t<=0xcf: sz=1<<(t-0xcc); return int.from_bytes(b[i+1:i+1+sz],'big'),i+1+sz
    if 0xd0<=t<=0xd3: sz=1<<(t-0xd0); return int.from_bytes(b[i+1:i+1+sz],'big',signed=True),i+1+sz
    if t in (0xd9,0xda,0xdb):
        sz={0xd9:1,0xda:2,0xdb:4}[t]; n=int.from_bytes(b[i+1:i+1+sz],'big'); return b[i+1+sz:i+1+sz+n].decode('utf-8','replace'),i+1+sz+n
    if t in (0xdc,0xdd):
        sz=2 if t==0xdc else 4; n=int.from_bytes(b[i+1:i+1+sz],'big'); return _arr(b,i+1+sz,n)
    if t in (0xde,0xdf):
        sz=2 if t==0xde else 4; n=int.from_bytes(b[i+1:i+1+sz],'big'); return _map(b,i+1+sz,n)
    if 0xd4<=t<=0xd8 or 0xc7<=t<=0xc9:
        raise ValueError('ext')
    raise ValueError('msgpack 0x%x'%t)
def _arr(b,i,n):
    out=[]
    for _ in range(n): v,i=mp_unpack(b,i); out.append(v)
    return out,i
def _map(b,i,n):
    out={}
    for _ in range(n):
        k,i=mp_unpack(b,i); v,i=mp_unpack(b,i)
        out[k if not isinstance(k,(list,dict)) else str(k)]=v
    return out,i
def elf_notes(e):
    e=bytes(e)
    shoff=struct.unpack_from('<Q',e,0x28)[0]; shentsize,shnum,shstrndx=struct.unpack_from('<HHH',e,0x3a)
    secs=[struct.unpack_from('<IIQQQQIIQQ',e,shoff+i*shentsize) for i in range(shnum)]
    out=[]
    for s in secs:
        if s[1]==7:  # SHT_NOTE
            off,size=s[4],s[5]; p=off
            while p+12<=off+size:
                nsz,dsz,ty=struct.unpack_from('<III',e,p); p+=12
                name=e[p:p+nsz].rstrip(b'\0'); p+=(nsz+3)&~3
                desc=e[p:p+dsz]; p+=(dsz+3)&~3
                out.append((name,ty,desc))
    syms=[]
    for s in secs:
        if s[1]==2:  # SYMTAB
            strtab=secs[s[6]]; so=strtab[4]
            for k in range(s[5]//24):
                nm,info,oth,shn,val,sz=struct.unpack_from('<IBBHQQ',e,s[4]+k*24)
                n=e[so+nm:e.index(b'\0',so+nm)].decode()
                syms.append((n,val,sz,shn))
    return out,syms,secs
def pipeline_meta(e):
    notes,syms,secs=elf_notes(e)
    for name,ty,desc in notes:
        if name==b'AMDGPU' and ty==32:
            m,_=mp_unpack(desc)
            return m,syms
    return None,syms
