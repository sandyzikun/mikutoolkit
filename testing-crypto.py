import mikutoolkit as miku
print(miku.crypto.phi.decode(miku.crypto.phi.encode("绫")))
print(miku.crypto.phi.decode(miku.crypto.phi.encode("綾")))
print(miku.crypto.phi.decode(miku.crypto.phi.encode("初音ミクV4Cu")))
for fn in [ "issenkou" , "melt" , "needle" ]:
    miku.crypto.phi.enfile("./test/test-%s.txt" % fn).close()
    miku.crypto.phi.defile("./test/test-%s.txt" % fn + ".mikucrypto").close()