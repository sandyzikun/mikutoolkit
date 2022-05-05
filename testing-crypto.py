import mikutoolkit as miku
print(miku.crypto.phi.decode(miku.crypto.phi.encode("初音ミクV4Cu")))
print(miku.crypto.phi.decode(miku.crypto.phi.encode("绫")))
print(miku.crypto.phi.decode(miku.crypto.phi.encode("綾")))
print("***")
for fn in [ "issenkou" , "melt" ]:
    miku.crypto.phi.enfile("test-%s.txt" % fn).close()
    miku.crypto.phi.defile("test-%s.txt" % fn + ".mikucrypto").close()